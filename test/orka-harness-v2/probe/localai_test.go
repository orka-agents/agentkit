package main

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestLocalAIRequestRemovesOnlyStrictFunctionHints(t *testing.T) {
	body := []byte(`{
		"model":"qwen-3.5-2b","stream":true,"tool_choice":"auto",
		"messages":[{"role":"user","content":"Do not call tools."}],
		"vendor_integer":9007199254740993,
		"tools":[
			{"type":"function","function":{"name":"echo","strict":true,"parameters":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":false}}},
			{"type":"function","function":{"name":"other","strict":false,"parameters":{"type":"object"}}},
			{"type":"web_search","strict":true}
		]
	}`)
	got, err := localAIRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	decode := func(data []byte) map[string]json.RawMessage {
		var value map[string]json.RawMessage
		if err := json.Unmarshal(data, &value); err != nil {
			t.Fatal(err)
		}
		return value
	}
	before, after := decode(body), decode(got)
	for _, key := range []string{"model", "stream", "tool_choice", "messages", "vendor_integer"} {
		if !reflect.DeepEqual(before[key], after[key]) {
			t.Fatalf("changed %s: %s != %s", key, after[key], before[key])
		}
	}
	var toolsBefore, toolsAfter []map[string]json.RawMessage
	if err := json.Unmarshal(before["tools"], &toolsBefore); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(after["tools"], &toolsAfter); err != nil {
		t.Fatal(err)
	}
	original, projected := decode(toolsBefore[0]["function"]), decode(toolsAfter[0]["function"])
	if _, exists := projected["strict"]; exists {
		t.Fatal("strict hint remained")
	}
	delete(original, "strict")
	// Normalize whitespace only; all function definition fields must survive.
	for key, value := range original {
		var a, b any
		if err := json.Unmarshal(value, &a); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(projected[key], &b); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(a, b) {
			t.Fatalf("changed function field %s", key)
		}
	}
	if !reflect.DeepEqual(toolsBefore[1:], toolsAfter[1:]) {
		t.Fatal("changed non-strict or non-function tools")
	}
}

func TestLocalAIRequestPreservesRequestsWithoutStrictTools(t *testing.T) {
	for _, body := range []string{`{"messages":[],"model":"qwen-3.5-2b"}`, `{"tools":[]}`, `{"tools":[{"type":"function","function":{"name":"echo","strict":false}}]}`} {
		got, err := localAIRequest([]byte(body))
		if err != nil || string(got) != body {
			t.Fatalf("unexpected request projection: %s, %v", got, err)
		}
	}
}

func TestLocalAIRequestRejectsMalformedTools(t *testing.T) {
	for _, body := range []string{`{`, `{"tools":{}}`, `{"tools":[{"type":"function","function":"invalid"}]}`} {
		if _, err := localAIRequest([]byte(body)); err == nil {
			t.Fatalf("accepted malformed request %s", body)
		}
	}
}
