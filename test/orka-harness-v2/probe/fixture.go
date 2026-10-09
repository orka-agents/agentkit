package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"time"

	v2 "github.com/orka-agents/orka/internal/harness/v2"
)

// Only the external provider and controller are fixtures. ACP, both supervisor
// proxies, MCP transports, and framework clients all run in the composed image.
type fixture struct {
	cfg      settings
	ctx      context.Context
	upstream string
	client   *http.Client
	mu       sync.Mutex
	current  *scenario
	provider int
	broker   int
	active   int
	faults   []string
}

type historyMessage struct {
	Role string
	Text string
}

type scenario struct {
	name         string
	kind         string
	input        string
	marker       string
	receipt      string
	previous     []historyMessage
	metadata     v2.MutationMetadata
	blocked      chan time.Time
	disconnected chan struct{}
	requests     int
	tools        int
	toolResults  int
	terminal     v2.EventType
}

type chatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCallID string          `json:"tool_call_id"`
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
	Tools    []struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	} `json:"tools"`
}

func startFixture(ctx context.Context, cfg settings, upstream string) (*fixture, func(), error) {
	f := &fixture{cfg: cfg, ctx: ctx, upstream: upstream, client: &http.Client{
		Transport:     v2.NewProxylessTransport(),
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", f.completion)
	mux.HandleFunc("POST "+v2.MCPBrokerCallPath, f.tool)
	listener, err := net.Listen("tcp", ":8090")
	if err != nil {
		return nil, nil, err
	}
	server := &http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 15 * time.Second, IdleTimeout: 30 * time.Second}
	go func() {
		if err := server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			f.fault("fixture HTTP server failed")
		}
	}()
	return f, func() { _ = server.Close(); f.client.CloseIdleConnections() }, nil
}

func (f *fixture) fault(message string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.faults = append(f.faults, message)
}

func (f *fixture) begin(provider bool) *scenario {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.active++
	if provider {
		f.provider++
	} else {
		f.broker++
	}
	return f.current
}

func (f *fixture) end() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.active--
}

func (f *fixture) selectScenario(s *scenario) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.active != 0 {
		return errors.New("previous fixture requests have not settled")
	}
	f.current = s
	return nil
}

func (f *fixture) counts() (provider, broker, active int, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.faults) > 0 {
		err = errors.New(strings.Join(f.faults, "; "))
	}
	return f.provider, f.broker, f.active, err
}

func (f *fixture) reject(w http.ResponseWriter, message string) {
	f.fault(message)
	writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]string{"message": "fixture request rejected", "type": "invalid_request_error"}})
}

func messageText(content json.RawMessage) string {
	var plain string
	if json.Unmarshal(content, &plain) == nil {
		return plain
	}
	var blocks []struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(content, &blocks) == nil {
		var text strings.Builder
		for _, block := range blocks {
			text.WriteString(block.Text)
		}
		return text.String()
	}
	return ""
}

func (f *fixture) completion(w http.ResponseWriter, r *http.Request) {
	s := f.begin(true)
	defer f.end()
	if s == nil || r.Header.Get("Authorization") != "Bearer "+f.cfg.ProviderToken {
		f.reject(w, "unexpected provider request or upstream credential")
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<20))
	if err != nil {
		f.reject(w, "provider request exceeds fixture bound")
		return
	}
	var request chatRequest
	if json.Unmarshal(body, &request) != nil || request.Model != f.cfg.Profile.Model {
		f.reject(w, "provider request model differs from the baked profile")
		return
	}
	var history []historyMessage
	toolResult := false
	for _, message := range request.Messages {
		text := messageText(message.Content)
		if (message.Role == "user" || message.Role == "assistant") && text != "" {
			history = append(history, historyMessage{message.Role, text})
		}
		if message.Role == "tool" && strings.Contains(text, s.receipt) {
			toolResult = true
		}
	}
	f.mu.Lock()
	s.requests++
	first := s.requests == 1
	if toolResult {
		s.toolResults++
	}
	f.mu.Unlock()
	if first {
		expected := append(append([]historyMessage{}, s.previous...), historyMessage{"user", s.input})
		if !reflect.DeepEqual(history, expected) {
			f.reject(w, "provider history must contain each successful user/assistant turn exactly once")
			return
		}
	}
	if s.kind == "provider-failure" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]string{
			"message": "injected E2E provider failure", "type": "invalid_request_error", "code": "e2e_failure",
		}})
		return
	}
	if f.upstream != "" {
		f.forward(w, r, body)
		return
	}
	message := map[string]any{"role": "assistant", "content": s.marker}
	finish := "stop"
	if s.kind == "tool" || s.kind == "tool-failure" || s.kind == "cancel" || s.kind == "deadline" {
		if toolResult {
			message["content"] = s.marker + " " + s.receipt
		} else {
			if !first || len(request.Tools) != 1 || request.Tools[0].Type != "function" || request.Tools[0].Function.Name != modelToolName {
				f.reject(w, "provider did not receive exactly the allowed MCP tool")
				return
			}
			arguments, _ := json.Marshal(map[string]string{"value": s.marker})
			message["content"] = nil
			message["tool_calls"] = []map[string]any{{
				"id": "call-" + randomID(), "type": "function",
				"function": map[string]string{"name": modelToolName, "arguments": string(arguments)},
			}}
			finish = "tool_calls"
		}
	}
	respondCompletion(w, request, message, finish)
}

func (f *fixture) forward(w http.ResponseWriter, r *http.Request, body []byte) {
	var err error
	body, err = localAIRequest(body)
	if err != nil {
		f.reject(w, "cannot construct AIKit provider request")
		return
	}
	upstream, err := http.NewRequestWithContext(r.Context(), http.MethodPost, strings.TrimRight(f.upstream, "/")+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		f.reject(w, "cannot construct live provider request")
		return
	}
	upstream.Header.Set("Content-Type", "application/json")
	// AIKit runs on the private test network and requires no provider credential.
	response, err := f.client.Do(upstream)
	if err != nil {
		f.reject(w, "live provider transport failed")
		return
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		f.reject(w, fmt.Sprintf("live provider returned HTTP %d", response.StatusCode))
		return
	}
	w.Header().Set("Content-Type", response.Header.Get("Content-Type"))
	w.WriteHeader(response.StatusCode)
	// Flush SSE chunks as they arrive; do not turn a real provider stream into
	// a buffered or synthetic answer.
	buffer := make([]byte, 16<<10)
	for {
		n, readErr := response.Body.Read(buffer)
		if n > 0 {
			if _, err := w.Write(buffer[:n]); err != nil {
				return
			}
			if flush, ok := w.(http.Flusher); ok {
				flush.Flush()
			}
		}
		if readErr != nil {
			if !errors.Is(readErr, io.EOF) && r.Context().Err() == nil {
				f.fault("live provider response stream failed")
			}
			return
		}
	}
}

func respondCompletion(w http.ResponseWriter, request chatRequest, message map[string]any, finish string) {
	id := "chatcmpl-" + randomID()
	base := map[string]any{"id": id, "object": "chat.completion", "created": time.Now().Unix(), "model": request.Model}
	if !request.Stream {
		base["choices"] = []any{map[string]any{"index": 0, "message": message, "finish_reason": finish}}
		base["usage"] = map[string]int{"prompt_tokens": 8, "completion_tokens": 8, "total_tokens": 16}
		writeJSON(w, http.StatusOK, base)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	base["object"] = "chat.completion.chunk"
	if calls, ok := message["tool_calls"].([]map[string]any); ok {
		for index, call := range calls {
			call["index"] = index
		}
	}
	base["choices"] = []any{map[string]any{"index": 0, "delta": message, "finish_reason": nil}}
	data, _ := json.Marshal(base)
	_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
	base["choices"] = []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": finish}}
	data, _ = json.Marshal(base)
	_, _ = fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", data)
	if flush, ok := w.(http.Flusher); ok {
		flush.Flush()
	}
}

func (f *fixture) tool(w http.ResponseWriter, r *http.Request) {
	s := f.begin(false)
	defer f.end()
	if s == nil || r.Header.Get("Authorization") != "Bearer "+f.cfg.ControllerToken ||
		r.Header.Get(v2.MCPBrokerPoolNamespaceHeader) != "e2e" || r.Header.Get(v2.MCPBrokerPoolUIDHeader) != string(f.cfg.Fence.RuntimePoolUID) {
		f.reject(w, "broker authentication or pool identity mismatch")
		return
	}
	var request v2.MCPBrokerCallRequest
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&request) != nil {
		f.reject(w, "broker request is not the native Orka shape")
		return
	}
	descriptor, err := request.ValidateAt(time.Now().UTC())
	if err != nil || request.Authorization.ValidateProfile(f.cfg.Profile) != nil ||
		v2.VerifyOperationCapability([]byte(f.cfg.CapabilityKey), r.Header.Get(v2.OperationCapabilityHeader), request.Metadata, true, time.Now().UTC()) != nil {
		f.reject(w, "broker request failed canonical policy/capability validation")
		return
	}
	if request.Namespace != "e2e" || request.Metadata.Fence != s.metadata.Fence || request.Metadata.TaskUID != s.metadata.TaskUID ||
		request.Metadata.TaskAttempt != s.metadata.TaskAttempt || request.Metadata.PromptID != s.metadata.PromptID ||
		request.Call.ToolName != toolName || descriptor.DefinitionDigest != f.cfg.MCP.ToolPolicy.Tools[0].DefinitionDigest || request.Call.Approval != nil {
		f.reject(w, "broker tool, prompt correlation, or registered definition mismatch")
		return
	}
	var arguments map[string]string
	if json.Unmarshal(request.Call.Arguments, &arguments) != nil || len(arguments) != 1 || arguments["value"] != s.marker {
		f.reject(w, "broker did not receive the expected tool arguments")
		return
	}
	f.mu.Lock()
	s.tools++
	f.mu.Unlock()
	if s.kind == "cancel" || s.kind == "deadline" {
		select {
		case s.blocked <- time.Now():
		default:
		}
		select {
		case <-r.Context().Done():
		case <-f.ctx.Done():
		}
		select {
		case s.disconnected <- struct{}{}:
		default:
		}
		return
	}
	if s.kind == "tool-failure" {
		// A broker transport failure must stop the run through the real MCP
		// error path, without a second injected failure at the provider.
		writeJSON(w, http.StatusServiceUnavailable, v2.ErrorResponse{
			Protocol: v2.ProtocolVersion, Code: v2.ErrorCodeOutcomeUnknown, Message: "injected E2E broker failure",
		})
		return
	}
	result, _ := json.Marshal(map[string]string{"receipt": s.receipt})
	writeJSON(w, http.StatusOK, v2.MCPBrokerCallResponse{
		Protocol: v2.ProtocolVersion, CallID: request.Call.CallID, Result: result,
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
