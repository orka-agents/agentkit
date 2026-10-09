package main

import "encoding/json"

// localAIRequest removes OpenAI's strict-schema hint from function tools. LocalAI
// v4.10.0 uses strict:true to force a tool-only grammar even with tool_choice:auto,
// overriding the model's disabled grammar and excluding ordinary text replies.
// This compatibility projection applies only to the AIKit live test provider;
// the tool schema, choice, messages, and actual generation remain unchanged.
func localAIRequest(body []byte) ([]byte, error) {
	var request map[string]json.RawMessage
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, err
	}
	if len(request["tools"]) == 0 {
		return body, nil
	}
	var tools []map[string]json.RawMessage
	if err := json.Unmarshal(request["tools"], &tools); err != nil {
		return nil, err
	}
	changed := false
	for _, tool := range tools {
		if string(tool["type"]) != `"function"` {
			continue
		}
		var function map[string]json.RawMessage
		if err := json.Unmarshal(tool["function"], &function); err != nil {
			return nil, err
		}
		if string(function["strict"]) != "true" {
			continue
		}
		delete(function, "strict")
		encoded, err := json.Marshal(function)
		if err != nil {
			return nil, err
		}
		tool["function"] = encoded
		changed = true
	}
	if !changed {
		return body, nil
	}
	encoded, err := json.Marshal(tools)
	if err != nil {
		return nil, err
	}
	request["tools"] = encoded
	return json.Marshal(request)
}
