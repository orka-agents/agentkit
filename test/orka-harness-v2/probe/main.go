// The runner copies this command into Orka's pinned module so every harness
// request, signature, digest, and event decoder uses the production contract.
package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	v2 "github.com/orka-agents/orka/internal/harness/v2"
)

const (
	toolName = "e2e_echo"
	// Each adapter prefixes tools with Orka's canonical MCP server name.
	modelToolName = "orka_" + toolName
)

type settings struct {
	Adapter         string
	Mode            string
	Profile         v2.RuntimeProfile
	MCP             v2.MCPPolicyConfiguration
	Fence           v2.Fence
	ControllerToken string
	CapabilityKey   string
	ProviderToken   string
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, 12*time.Minute)
	defer cancel()
	if err := command(ctx, os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "v2 E2E:", err)
		os.Exit(1)
	}
}

func command(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return errors.New("expected prepare or run")
	}
	flags := flag.NewFlagSet(args[0], flag.ContinueOnError)
	dir := flags.String("dir", "/e2e", "private fixture directory")
	adapter := flags.String("adapter", "", "AgentKit adapter")
	adapterDigest := flags.String("adapter-digest", "", "immutable AgentKit source image digest")
	model := flags.String("model", "", "baked model name")
	mode := flags.String("mode", "offline", "offline or live")
	runtimeURL := flags.String("runtime-url", "http://runtime:8080", "production supervisor URL")
	results := flags.String("results", "/results/result.json", "sanitized result file")
	upstream := flags.String("upstream", "", "live OpenAI-compatible provider URL")
	if err := flags.Parse(args[1:]); err != nil {
		return err
	}
	switch args[0] {
	case "prepare":
		return prepare(*dir, *adapter, *adapterDigest, *model, *mode)
	case "run":
		data, err := os.ReadFile(filepath.Join(*dir, "settings.json"))
		if err != nil {
			return err
		}
		var cfg settings
		if err := json.Unmarshal(data, &cfg); err != nil {
			return err
		}
		if (cfg.Mode == "live") != (*upstream != "") {
			return errors.New("live mode requires a provider upstream; offline mode forbids it")
		}
		return exercise(ctx, cfg, *runtimeURL, *upstream, *results)
	default:
		return errors.New("expected prepare or run")
	}
}

func sha(data []byte) string {
	digest := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func randomID() string {
	// crypto/rand.Read fills the buffer or terminates on an OS entropy failure.
	var data [16]byte
	_, _ = rand.Read(data[:])
	return hex.EncodeToString(data[:])
}

func prepare(dir, adapter, adapterDigest, model, mode string) error {
	if adapter == "" || model == "" || (mode != "offline" && mode != "live") {
		return errors.New("prepare requires adapter, model, and offline/live mode")
	}
	config, err := os.ReadFile(filepath.Join(dir, "agent.yaml"))
	if err != nil {
		return err
	}
	tools := []v2.MCPToolDescriptor{{
		Name: toolName, Description: "Return a receipt for the supplied value. Call this when asked to use e2e_echo.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":false}`),
		Source:      v2.MCPToolSourceBrokeredCustom, Effect: v2.MCPToolEffectReadOnly,
		DefinitionDigest: sha([]byte("agentkit-v2-e2e-echo-v1")),
	}}
	policy := v2.MCPPolicyConfiguration{ToolPolicy: v2.MCPToolPolicy{
		AllowedToolNames: []string{toolName}, DisallowedToolNames: []string{}, Tools: tools,
	}}
	policy.ToolPolicy.DescriptorDigest, err = v2.CanonicalMCPToolDescriptorDigest(tools)
	if err != nil {
		return err
	}
	policy.ToolPolicyDigest, err = v2.CanonicalRuntimeToolPolicyDigest(policy.ToolPolicy.AllowedToolNames, policy.ToolPolicy.DisallowedToolNames, false)
	if err != nil {
		return err
	}
	policy.ApprovalPolicyDigest, err = v2.CanonicalMCPApprovalPolicyDigest(policy.ApprovalPolicy)
	if err != nil {
		return err
	}
	policy.MCPConfigurationDigest, err = v2.CanonicalMCPConfigurationDigest(policy.ToolPolicy.AllowedToolNames)
	if err != nil {
		return err
	}
	profile := v2.RuntimeProfile{
		ACPProfile: v2.ACPProfileV1, ProviderKind: "agentkit", Model: model,
		AdapterDigests:           map[string]string{"agentkit-serve-acp": adapterDigest},
		AgentConfigurationDigest: sha(config), ToolPolicyDigest: policy.ToolPolicyDigest,
		ApprovalPolicyDigest: policy.ApprovalPolicyDigest, MCPConfigurationDigest: policy.MCPConfigurationDigest,
		WorkspaceIntent: v2.WorkspaceIntentRead, ProxyCredentialRole: "e2e-provider",
		ProxyCredentialScope: "e2e", ResourceClass: "standard",
	}
	if err := profile.Validate(); err != nil {
		return err
	}
	if err := policy.ValidateProfile(profile); err != nil {
		return err
	}
	profileDigest, err := v2.CanonicalProfileDigest(profile)
	if err != nil {
		return err
	}
	identity := randomID()
	cfg := settings{
		Adapter: adapter, Mode: mode, Profile: profile, MCP: policy,
		ControllerToken: randomID() + randomID(), CapabilityKey: randomID() + randomID(), ProviderToken: randomID(),
		Fence: v2.Fence{
			RuntimeInstanceID: v2.RuntimeInstanceID("runtime-" + identity), SupervisorBootID: v2.SupervisorBootID("boot-" + identity),
			ControllerEpoch: 1, RuntimePoolUID: v2.RuntimePoolUID("pool-" + identity), RuntimePoolGeneration: 1,
			RuntimeProfileDigest: profileDigest, ProfileDigestSchemaVersion: v2.ProfileDigestSchemaVersion,
		},
	}
	secretDir := filepath.Join(dir, "secrets")
	if err := os.MkdirAll(secretDir, 0o700); err != nil {
		return err
	}
	for name, value := range map[string]string{"controller": cfg.ControllerToken, "capability": cfg.CapabilityKey, "provider": cfg.ProviderToken} {
		if err := os.WriteFile(filepath.Join(secretDir, name), []byte(value), 0o600); err != nil {
			return err
		}
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "settings.json"), data, 0o600); err != nil {
		return err
	}
	env := map[string]string{
		"ORKA_ACP_LISTEN_ADDRESS": ":8080", "ORKA_ACP_PROVIDER": profile.ProviderKind,
		"ORKA_ACP_MODEL": model, "ORKA_ACP_WORKSPACE_INTENT": string(profile.WorkspaceIntent),
		"ORKA_ACP_AGENT_CONFIGURATION_DIGEST": profile.AgentConfigurationDigest,
		"ORKA_ACP_TOOL_POLICY_DIGEST":         policy.ToolPolicyDigest, "ORKA_ACP_APPROVAL_POLICY_DIGEST": policy.ApprovalPolicyDigest,
		"ORKA_ACP_MCP_CONFIGURATION_DIGEST": policy.MCPConfigurationDigest,
		"ORKA_ACP_PROXY_CREDENTIAL_ROLE":    profile.ProxyCredentialRole, "ORKA_ACP_PROXY_CREDENTIAL_SCOPE": profile.ProxyCredentialScope,
		"ORKA_ACP_RESOURCE_CLASS": profile.ResourceClass, "ORKA_ACP_CONTROLLER_EPOCH": "1", "ORKA_ACP_RUNTIME_POOL_GENERATION": "1",
		"ORKA_ACP_RUNTIME_INSTANCE_ID": string(cfg.Fence.RuntimeInstanceID), "ORKA_ACP_SUPERVISOR_BOOT_ID": string(cfg.Fence.SupervisorBootID),
		"ORKA_ACP_RUNTIME_POOL_UID": string(cfg.Fence.RuntimePoolUID), "ORKA_ACP_SESSION_BASE_DIR": "/sessions",
		"ORKA_ACP_PROVIDER_PROXY_BASE_URL": "http://fixture:8090/v1", "ORKA_ACP_MCP_BROKER_URL": "http://fixture:8090",
		"ORKA_ACP_TRUST_NAMESPACE": "e2e", "ORKA_ACP_CONTROLLER_TOKEN_FILE": "/e2e/secrets/controller",
		"ORKA_ACP_CAPABILITY_SECRET_FILE": "/e2e/secrets/capability", "ORKA_ACP_PROVIDER_TOKEN_FILE": "/e2e/secrets/provider",
	}
	var lines []string
	for name, value := range env {
		if strings.ContainsAny(value, "\r\n") {
			return fmt.Errorf("invalid environment value for %s", name)
		}
		lines = append(lines, name+"="+value)
	}
	sort.Strings(lines)
	return os.WriteFile(filepath.Join(dir, "runtime.env"), []byte(strings.Join(lines, "\n")+"\n"), 0o644)
}
