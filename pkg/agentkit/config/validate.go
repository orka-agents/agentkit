package config

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/big"
	pathpkg "path"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/orka-agents/agentkit/pkg/agentkit/runtimes"
	"github.com/orka-agents/agentkit/pkg/utils"
)

const (
	ContextTypeSearch       = "search"
	ContextTypeSkills       = "skills"
	ContextTypeMemory       = "memory"
	ContextSourceFilesystem = "filesystem"
	ContextSourceMCP        = "mcp"

	ToolTypeMCP                 = "mcp"
	ToolTransportStdio          = "stdio"
	ToolTransportStreamableHTTP = "streamable-http"
	AuthTypeBearer              = "bearer"
	AuthTypeWorkloadIdentity    = "workload-identity-token"
	ApprovalNever               = "never"
	ApprovalAuto                = "auto"
	ApprovalAlways              = "always"

	BrokeredClassRead         = "read"
	BrokeredClassWrite        = "write"
	BrokeredClassCoordination = "coordination"

	jsonSchemaTypeKey              = "type"
	jsonSchemaTypeObject           = "object"
	jsonSchemaTypeNumber           = "number"
	jsonSchemaTypeString           = "string"
	jsonSchemaTypeInteger          = "integer"
	jsonSchemaTypeArray            = "array"
	jsonSchemaTypeBoolean          = "boolean"
	jsonSchemaTypeNull             = "null"
	jsonSchemaEnumKey              = "enum"
	jsonSchemaDefaultKey           = "default"
	jsonSchemaRequiredKey          = "required"
	jsonSchemaDependentRequiredKey = "dependentRequired"
	jsonSchemaMinimumKey           = "minimum"
	jsonSchemaMaximumKey           = "maximum"
	brokeredDigestDescriptionKey   = "description"
	brokeredDigestNumberKey        = "\u0000agentkit_json_number"
	brokeredAuthenticationWord     = "authentication"
	brokeredSensitiveWord          = "credential"
	brokeredSensitivePluralWord    = "credentials"
	brokeredTokenWord              = "token"
	brokeredTokensWord             = "tokens"
	brokeredUnsafeCookieKey        = "cookie"
	authorizationKey               = "authorization"
	credentialHeaderAPIKey         = "api-key"
	maxExactJSONFloatInteger       = float64(1<<53 - 1)
)

var brokeredBasicValuePattern = regexp.MustCompile(`(?i)(?:^|[^A-Za-z0-9_])basic[^A-Za-z0-9_]+?([A-Za-z0-9+/]+={0,2})`)

// Brokered descriptions use the Unicode 15 casing contract shared with the
// Python ABI reader. Exclude the mappings introduced in Unicode 16 and 17;
// the full-scalar golden test detects further drift when Go is upgraded.
var brokeredUnicode15Case = unicode.SpecialCase{
	{Lo: 0x1C89, Hi: 0x1C89},
	{Lo: 0xA7CB, Hi: 0xA7CC},
	{Lo: 0xA7CE, Hi: 0xA7CE},
	{Lo: 0xA7D2, Hi: 0xA7D2},
	{Lo: 0xA7D4, Hi: 0xA7D4},
	{Lo: 0xA7DA, Hi: 0xA7DA},
	{Lo: 0xA7DC, Hi: 0xA7DC},
	{Lo: 0x10D50, Hi: 0x10D65},
	{Lo: 0x16EA0, Hi: 0x16EB8},
}

// Validate reports every problem with the config at once via errors.Join (plan
// §16.2 #3 — one report-all validator, not scattered first-error-wins funcs).
//
// It covers both schema validity and the v0 deterministic security gates that
// run BEFORE any LLB (plan §10): no secret literals in YAML, exposure sanity.
// Behavioral evals are out of v0 scope; these static gates are the v0 safety net.
func (c *AgentConfig) Validate() error {
	var errs []error
	add := func(format string, args ...any) {
		errs = append(errs, fmt.Errorf(format, args...))
	}

	// --- apiVersion / kind -------------------------------------------------
	if c.APIVersion == "" {
		add("apiVersion is not defined")
	} else if c.APIVersion != utils.APIv1alpha1 {
		add("apiVersion %q is not supported (expected %q)", c.APIVersion, utils.APIv1alpha1)
	}
	if c.Kind != utils.KindAgent {
		add("kind %q is not supported (expected %q)", c.Kind, utils.KindAgent)
	}

	// --- metadata ----------------------------------------------------------
	if c.Metadata.Name == "" {
		add("metadata.name is required")
	}
	validateMetadataLabels(add, c.Metadata.Labels)

	// --- runtime -----------------------------------------------------------
	runtimeName := c.Runtime
	if runtimeName == "" {
		runtimeName = runtimes.DefaultRuntime()
	}
	runtimeSpec, runtimeKnown := runtimes.RuntimeByName(runtimeName)
	if c.Runtime != "" && !runtimeKnown {
		supported := runtimes.KnownRuntimes()
		sort.Strings(supported)
		add("runtime %q is not supported (supported: %s)", c.Runtime, strings.Join(supported, ", "))
	}

	// --- model -------------------------------------------------------------
	if c.Model.Provider == "" {
		add("model.provider is required")
	} else if c.Model.Provider != utils.ProviderOpenAICompatible {
		add("model.provider %q is not supported in v0 (only %q); local models compose with a co-located AIKit container over baseURL, never baked", c.Model.Provider, utils.ProviderOpenAICompatible)
	}
	if c.Model.BaseURL == "" {
		add("model.baseURL is required")
	}
	if c.Model.Name == "" {
		add("model.name is required")
	}
	// noSecretsInImage gate (plan §10): apiKeyEnv must be a NAME, not a literal.
	if c.Model.APIKeyEnv != "" && (!isEnvVarName(c.Model.APIKeyEnv) || looksLikeSecretLiteral(c.Model.APIKeyEnv)) {
		add("model.apiKeyEnv %q must be an env var NAME matching [A-Z0-9_]+; provide the NAME of an env var (e.g. OPENAI_API_KEY) and inject the value with `docker run -e`", c.Model.APIKeyEnv)
	}
	validateAuth(add, "model.auth", c.Model.Auth)
	if c.Model.Auth != nil && c.Model.Auth.Type != AuthTypeWorkloadIdentity {
		add("model.auth.type %q is not supported in v0; use apiKeyEnv or workload-identity-token gated by runtime capability", c.Model.Auth.Type)
	}

	// --- instructions ------------------------------------------------------
	if c.Instructions.IsZero() {
		add("instructions are required (a string, or a {file: ...} source)")
	} else if set := c.Instructions.variantsSet(); len(set) > 1 {
		add("instructions set multiple sources %v; exactly one of inline|file is allowed", set)
	}

	// --- tools -------------------------------------------------------------
	seen := map[string]bool{}
	for i, t := range c.Tools {
		if t.Name == "" {
			add("tools[%d].name is required", i)
		} else if seen[t.Name] {
			add("tools[%d]: duplicate tool name %q", i, t.Name)
		}
		seen[t.Name] = true

		if t.Type != "" && t.Type != ToolTypeMCP {
			add("tools[%d] (%s): type %q is not supported (expected %q)", i, t.Name, t.Type, ToolTypeMCP)
		}

		set := t.variantsSet()
		switch len(set) {
		case 0:
			add("tools[%d] (%s): a tool source is required (command or urlEnv)", i, t.Name)
		case 1:
			if t.Image != "" {
				add("tools[%d] (%s): image-based MCP servers are not supported in v0 (use command or streamable-http urlEnv; arbitrary-OCI staging is v1)", i, t.Name)
			}
		default:
			add("tools[%d] (%s): sets multiple sources %v; exactly one is allowed", i, t.Name, set)
		}

		validateToolTransport(add, i, t)
		validateToolEnv(add, i, t)
		validateToolHeaders(add, i, t)
		validateAuth(add, fmt.Sprintf("tools[%d] (%s).auth", i, t.Name), t.Auth)
		validateApproval(add, i, t)
	}

	validateBrokeredTools(add, c.BrokeredTools, seen)

	// --- env requirements ---------------------------------------------------
	seenEnv := map[string]bool{}
	for i, e := range c.Env {
		if e.Name == "" {
			add("env[%d].name is required; list env var NAMES only", i)
			continue
		}
		if !isEnvVarName(e.Name) || looksLikeSecretLiteral(e.Name) {
			add("env[%d].name %q must be an env var NAME matching [A-Z0-9_]+; list names only, never values", i, e.Name)
		}
		if seenEnv[e.Name] {
			add("env[%d]: duplicate env var name %q", i, e.Name)
		}
		seenEnv[e.Name] = true
	}

	validateContext(add, c.Context, c.Tools)
	validateObservability(add, c.Observability)

	// --- runtime capability gate -------------------------------------------
	if runtimeKnown {
		missing := runtimeSpec.MissingCapabilities(c.requiredCapabilities())
		if len(missing) > 0 {
			add("runtime %q does not support requested capabilities: %s", runtimeSpec.Name, strings.Join(missing, ", "))
		}
	}

	// --- expose ------------------------------------------------------------
	if !c.Expose.OpenAI {
		add("expose.openai must be true in v0 (the OpenAI /v1 façade is the only serving surface)")
	}
	if c.Expose.Port < 0 || c.Expose.Port > 65535 {
		add("expose.port %d is out of range", c.Expose.Port)
	}

	return errors.Join(errs...)
}

func validateBrokeredTools(add func(string, ...any), tools []BrokeredTool, ownedToolNames map[string]bool) {
	if len(tools) > 0 && len(ownedToolNames) > 0 {
		add("tools and brokeredTools cannot be mixed in v0; direct AgentKit-owned tools are disabled in brokered Foundry mode")
	}
	seen := map[string]bool{}
	for i, tool := range tools {
		path := fmt.Sprintf("brokeredTools[%d]", i)
		if tool.Name == "" {
			add("%s.name is required", path)
		} else {
			if !isBrokeredToolName(tool.Name) {
				add("%s.name %q must match [A-Za-z0-9_-]{1,64}", path, tool.Name)
			}
			if seen[tool.Name] {
				add("%s: duplicate brokered tool name %q", path, tool.Name)
			}
			if ownedToolNames[tool.Name] {
				add("%s.name %q cannot be both owned and brokered", path, tool.Name)
			}
			seen[tool.Name] = true
		}
		if tool.Description == "" {
			add("%s.description is required", path)
		} else if hasUnsafeBrokeredDescription(tool.Description) {
			add("%s.description must not contain URLs or secret-like material", path)
		}
		switch tool.BrokeredClass {
		case BrokeredClassRead, BrokeredClassWrite, BrokeredClassCoordination:
		case "":
			add("%s.brokeredClass is required", path)
		default:
			add("%s.brokeredClass %q is not supported (expected read, write, or coordination)", path, tool.BrokeredClass)
		}
		validateBrokeredToolParameters(add, path+".parameters", tool.Parameters)
		if tool.SchemaDigest != "" {
			if !isSchemaDigest(tool.SchemaDigest) {
				add("%s.schemaDigest must be sha256:<64 lowercase hex>", path)
			} else if actual, err := BrokeredToolSchemaDigest(tool); err != nil {
				add("%s.schemaDigest could not be checked: %v", path, err)
			} else if tool.SchemaDigest != actual {
				add("%s.schemaDigest does not match the safe schema", path)
			}
		}
	}
}

func validateBrokeredToolParameters(add func(string, ...any), path string, parameters map[string]any) {
	if parameters == nil {
		add("%s must be a JSON Schema object", path)
		return
	}
	if _, err := json.Marshal(parameters); err != nil {
		add("%s must be JSON serializable: %v", path, err)
		return
	}
	normalized, ok := normalizeJSONContainers(parameters).(map[string]any)
	if !ok {
		add("%s must be a JSON Schema object", path)
		return
	}
	canonical, err := canonicalJSON(normalized)
	if err != nil {
		add("%s must be JSON serializable: %v", path, err)
		return
	}
	if len(canonical) > 64*1024 {
		add("%s schema is too large", path)
	}
	schema := normalized
	validateJSONSchemaSubset(add, path, schema)
	if typ, _ := schema[jsonSchemaTypeKey].(string); typ != jsonSchemaTypeObject {
		add("%s must set type: object", path)
	}
	rejectUnsafeBrokeredSchemaKeys(add, path, schema)
	validateBrokeredSchemaValueConstraints(add, path, schema)
}

func normalizeJSONContainers(value any) any {
	if value == nil {
		return nil
	}
	rv := reflect.ValueOf(value)
	for rv.Kind() == reflect.Interface || rv.Kind() == reflect.Pointer {
		if rv.IsNil() {
			return nil
		}
		rv = rv.Elem()
	}
	if number, ok := rv.Interface().(json.Number); ok {
		return number
	}
	switch rv.Kind() {
	case reflect.Map:
		if rv.IsNil() {
			return nil
		}
		if rv.Type().Key().Kind() != reflect.String {
			return value
		}
		out := make(map[string]any, rv.Len())
		iter := rv.MapRange()
		for iter.Next() {
			out[iter.Key().String()] = normalizeJSONContainers(iter.Value().Interface())
		}
		return out
	case reflect.Slice:
		if rv.IsNil() {
			return nil
		}
		if rv.Type().Elem().Kind() == reflect.Uint8 {
			return value
		}
		out := make([]any, rv.Len())
		for i := 0; i < rv.Len(); i++ {
			out[i] = normalizeJSONContainers(rv.Index(i).Interface())
		}
		return out
	case reflect.Array:
		out := make([]any, rv.Len())
		for i := 0; i < rv.Len(); i++ {
			out[i] = normalizeJSONContainers(rv.Index(i).Interface())
		}
		return out
	case reflect.Bool:
		return rv.Bool()
	case reflect.String:
		return rv.String()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rv.Int()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return rv.Uint()
	case reflect.Float32:
		return float32(rv.Float())
	case reflect.Float64:
		return rv.Float()
	default:
		return rv.Interface()
	}
}

func hasAnySchemaKey(schema map[string]any, keys ...string) bool {
	for _, key := range keys {
		if _, ok := schema[key]; ok {
			return true
		}
	}
	return false
}

func validateJSONSchemaSubset(add func(string, ...any), path string, schema map[string]any) {
	for _, key := range []string{"allOf", "anyOf", "oneOf", "not", "$ref", "if", "then", "else", "contains", "minContains", "maxContains", "propertyNames", "dependentSchemas", "patternProperties", "unevaluatedProperties", "unevaluatedItems", "prefixItems", "uniqueItems"} {
		if _, ok := schema[key]; ok {
			add("%s.%s is not supported for deterministic brokered tool schemas", path, key)
		}
	}
	if value, ok := schema[jsonSchemaTypeKey]; ok {
		validateJSONSchemaType(add, path, value)
	}
	if properties, ok := schema["properties"]; ok {
		props, ok := properties.(map[string]any)
		if !ok {
			add("%s.properties must be an object", path)
		} else {
			for name, child := range props {
				childSchema, ok := child.(map[string]any)
				if !ok {
					add("%s.properties.%s must be a JSON Schema object", path, name)
					continue
				}
				validateJSONSchemaSubset(add, path+".properties."+name, childSchema)
			}
		}
	}
	if items, ok := schema["items"]; ok {
		switch typed := items.(type) {
		case map[string]any:
			validateJSONSchemaSubset(add, path+".items", typed)
		case []any:
			add("%s.items array form is not supported for brokered tool schemas", path)
		default:
			add("%s.items must be an object", path)
		}
	}
	if required, ok := schema[jsonSchemaRequiredKey]; ok && !isStringArray(required) {
		add("%s.required must be a string array", path)
	}
	if dependentRequired, ok := schema[jsonSchemaDependentRequiredKey]; ok {
		values, ok := dependentRequired.(map[string]any)
		if !ok {
			add("%s.dependentRequired must be an object", path)
		} else {
			for name, value := range values {
				if !isStringArray(value) {
					add("%s.dependentRequired.%s must be a string array", path, name)
				}
			}
		}
	}
	if enumValue, ok := schema[jsonSchemaEnumKey]; ok {
		items, ok := enumValue.([]any)
		if !ok {
			add("%s.enum must be an array", path)
		} else if len(items) == 0 {
			add("%s.enum must contain at least one value", path)
		}
	}
	if _, ok := schema["pattern"]; ok {
		add("%s.pattern is not supported for deterministic brokered tool schemas", path)
	}
	if additional, ok := schema["additionalProperties"]; ok {
		switch typed := additional.(type) {
		case bool:
		case map[string]any:
			validateJSONSchemaSubset(add, path+".additionalProperties", typed)
		default:
			add("%s.additionalProperties must be a boolean or object", path)
		}
	}
	if hasAnySchemaKey(schema, jsonSchemaEnumKey, "const", "default") && hasAnySchemaKey(schema, jsonSchemaMinimumKey, "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties", "pattern") {
		add("%s combines enum/const/default with constraints unsupported by deterministic brokered synthesis", path)
	}
	if _, ok := schema["multipleOf"]; ok {
		add("%s.multipleOf is not supported for deterministic brokered tool schemas", path)
	}
	for _, key := range []string{jsonSchemaMinimumKey, jsonSchemaMaximumKey, "exclusiveMinimum", "exclusiveMaximum"} {
		if value, ok := schema[key]; ok {
			if !isFiniteJSONNumber(value) {
				add("%s.%s must be a number", path, key)
			}
		}
	}
	for _, key := range []string{"minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"} {
		if value, ok := schema[key]; ok {
			if !isNonNegativeJSONInteger(value) {
				add("%s.%s must be a non-negative integer", path, key)
			}
		}
	}
}

func isFiniteJSONNumber(value any) bool {
	switch typed := value.(type) {
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return true
	case float32:
		return !math.IsNaN(float64(typed)) && !math.IsInf(float64(typed), 0)
	case float64:
		return !math.IsNaN(typed) && !math.IsInf(typed, 0)
	case json.Number:
		number, ok := parseJSONNumber(typed)
		return ok && jsonNumberIsRepresentable(typed, number)
	default:
		return false
	}
}

func isNonNegativeJSONInteger(value any) bool {
	switch typed := value.(type) {
	case int:
		return typed >= 0
	case int8:
		return typed >= 0
	case int16:
		return typed >= 0
	case int32:
		return typed >= 0
	case int64:
		return typed >= 0
	case uint, uint8, uint16, uint32, uint64:
		return true
	case float32:
		number := float64(typed)
		return !math.IsNaN(number) && !math.IsInf(number, 0) && number >= 0 && math.Trunc(number) == number
	case float64:
		return !math.IsNaN(typed) && !math.IsInf(typed, 0) && typed >= 0 && math.Trunc(typed) == typed
	case json.Number:
		number, ok := parseJSONNumber(typed)
		return ok && number.Sign() >= 0 && jsonNumberIsExactInteger(typed, number)
	default:
		return false
	}
}

func parseJSONNumber(value json.Number) (*big.Rat, bool) {
	raw := value.String()
	if !json.Valid([]byte(raw)) {
		return nil, false
	}
	return new(big.Rat).SetString(raw)
}

func jsonNumberIsExactInteger(value json.Number, number *big.Rat) bool {
	return number.IsInt() && jsonNumberIsRepresentable(value, number)
}

func jsonNumberIsRepresentable(value json.Number, number *big.Rat) bool {
	raw := value.String()
	if !strings.ContainsAny(raw, ".eE") {
		return true
	}
	parsed, err := strconv.ParseFloat(raw, 64)
	if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
		return false
	}
	roundTrip, ok := new(big.Rat).SetString(strconv.FormatFloat(parsed, 'g', -1, 64))
	return ok && roundTrip.Cmp(number) == 0
}

func validateJSONSchemaType(add func(string, ...any), path string, value any) {
	if value == nil {
		return
	}
	valid := func(v string) bool {
		switch v {
		case jsonSchemaTypeObject, jsonSchemaTypeString, jsonSchemaTypeInteger, jsonSchemaTypeNumber, jsonSchemaTypeBoolean, jsonSchemaTypeArray, jsonSchemaTypeNull:
			return true
		default:
			return false
		}
	}
	switch typed := value.(type) {
	case string:
		if !valid(typed) {
			add("%s.type %q is not supported", path, typed)
		}
	case []any:
		if len(typed) == 0 {
			add("%s.type must not be empty", path)
		}
		for _, item := range typed {
			text, ok := item.(string)
			if !ok {
				add("%s.type must contain only strings", path)
				continue
			}
			if !valid(text) {
				add("%s.type %q is not supported", path, text)
			}
		}
	default:
		add("%s.type must be a string or string array", path)
	}
}

func validateBrokeredSchemaValueConstraints(add func(string, ...any), path string, value any) {
	schema, ok := value.(map[string]any)
	if !ok {
		return
	}
	types, ok := brokeredSchemaTypes(add, path, schema)
	if ok && len(types) > 0 {
		for _, keyword := range []string{"const", jsonSchemaDefaultKey} {
			if child, exists := schema[keyword]; exists && !matchesAnySchemaType(child, types) {
				add("%s.%s must match the declared JSON Schema type", path, keyword)
			}
		}
		if enum, exists := schema[jsonSchemaEnumKey]; exists {
			items, ok := enum.([]any)
			if !ok {
				add("%s.enum must be an array", path)
			} else {
				for i, item := range items {
					if !matchesAnySchemaType(item, types) {
						add("%s.enum[%d] must match the declared JSON Schema type", path, i)
					}
				}
			}
		}
	}
	if properties, ok := schema["properties"].(map[string]any); ok {
		for name, child := range properties {
			if childSchema, ok := child.(map[string]any); ok {
				validateBrokeredSchemaValueConstraints(add, path+".properties."+name, childSchema)
			}
		}
	}
	if items, ok := schema["items"].(map[string]any); ok {
		validateBrokeredSchemaValueConstraints(add, path+".items", items)
	} else if tupleItems, ok := schema["items"].([]any); ok {
		for i, child := range tupleItems {
			if childSchema, ok := child.(map[string]any); ok {
				validateBrokeredSchemaValueConstraints(add, fmt.Sprintf("%s.items[%d]", path, i), childSchema)
			}
		}
	}
	if additional, ok := schema["additionalProperties"].(map[string]any); ok {
		validateBrokeredSchemaValueConstraints(add, path+".additionalProperties", additional)
	}
}

func brokeredSchemaTypes(add func(string, ...any), path string, schema map[string]any) ([]string, bool) {
	raw, exists := schema[jsonSchemaTypeKey]
	if !exists {
		return nil, true
	}
	switch typed := raw.(type) {
	case string:
		if !isSupportedBrokeredSchemaType(typed) {
			add("%s.type %q is not supported", path, typed)
			return nil, false
		}
		return []string{typed}, true
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			name, ok := item.(string)
			if !ok || !isSupportedBrokeredSchemaType(name) {
				add("%s.type must be a string or string array", path)
				return nil, false
			}
			out = append(out, name)
		}
		return out, true
	default:
		add("%s.type must be a string or string array", path)
		return nil, false
	}
}

func isSupportedBrokeredSchemaType(schemaType string) bool {
	switch schemaType {
	case "null", "boolean", "integer", jsonSchemaTypeNumber, jsonSchemaTypeString, "array", jsonSchemaTypeObject:
		return true
	default:
		return false
	}
}

func matchesAnySchemaType(value any, types []string) bool {
	for _, schemaType := range types {
		if matchesSchemaType(value, schemaType) {
			return true
		}
	}
	return false
}

func matchesSchemaType(value any, schemaType string) bool {
	switch schemaType {
	case jsonSchemaTypeNull:
		return value == nil
	case jsonSchemaTypeBoolean:
		_, ok := value.(bool)
		return ok
	case jsonSchemaTypeInteger:
		switch typed := value.(type) {
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
			return true
		case float64:
			return !math.IsNaN(typed) && !math.IsInf(typed, 0) && typed == math.Trunc(typed) && math.Abs(typed) <= maxExactJSONFloatInteger
		case json.Number:
			number, ok := parseJSONNumber(typed)
			return ok && jsonNumberIsExactInteger(typed, number)
		default:
			return false
		}
	case jsonSchemaTypeNumber:
		switch typed := value.(type) {
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
			return true
		case float32:
			return !math.IsNaN(float64(typed)) && !math.IsInf(float64(typed), 0)
		case float64:
			return !math.IsNaN(typed) && !math.IsInf(typed, 0)
		case json.Number:
			number, ok := parseJSONNumber(typed)
			return ok && jsonNumberIsRepresentable(typed, number)
		default:
			return false
		}
	case jsonSchemaTypeString:
		_, ok := value.(string)
		return ok
	case jsonSchemaTypeArray:
		_, ok := value.([]any)
		return ok
	case jsonSchemaTypeObject:
		_, ok := value.(map[string]any)
		return ok
	default:
		return false
	}
}

func isStringArray(value any) bool {
	switch typed := value.(type) {
	case []string:
		return true
	case []any:
		for _, item := range typed {
			if _, ok := item.(string); !ok {
				return false
			}
		}
		return true
	default:
		return false
	}
}

func rejectUnsafeBrokeredSchemaKeys(add func(string, ...any), path string, value any) {
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			childPath := path + "." + key
			if isUnsafeBrokeredKey(key) {
				add("%s is not safe for brokered tool schemas", childPath)
			}
			if key == jsonSchemaRequiredKey || key == jsonSchemaDependentRequiredKey {
				rejectUnsafeBrokeredPropertyNameValues(add, childPath, child)
			}
			rejectUnsafeBrokeredSchemaKeys(add, childPath, child)
		}
	case []any:
		for i, child := range typed {
			rejectUnsafeBrokeredSchemaKeys(add, fmt.Sprintf("%s[%d]", path, i), child)
		}
	case []map[string]any:
		for i, child := range typed {
			rejectUnsafeBrokeredSchemaKeys(add, fmt.Sprintf("%s[%d]", path, i), child)
		}
	case string:
		if hasUnsafeBrokeredText(typed) {
			add("%s contains URL or secret-like material", path)
		}
	}
}

func rejectUnsafeBrokeredPropertyNameValues(add func(string, ...any), path string, value any) {
	switch typed := value.(type) {
	case string:
		if isUnsafeBrokeredKey(typed) {
			add("%s value %q is not safe for brokered tool schemas", path, typed)
		}
	case []any:
		for i, child := range typed {
			rejectUnsafeBrokeredPropertyNameValues(add, fmt.Sprintf("%s[%d]", path, i), child)
		}
	case []string:
		for i, child := range typed {
			rejectUnsafeBrokeredPropertyNameValues(add, fmt.Sprintf("%s[%d]", path, i), child)
		}
	case map[string]any:
		for key, child := range typed {
			childPath := path + "." + key
			if isUnsafeBrokeredKey(key) {
				add("%s is not safe for brokered tool schemas", childPath)
			}
			rejectUnsafeBrokeredPropertyNameValues(add, childPath, child)
		}
	}
}

func isBrokeredToolName(value string) bool {
	if len(value) == 0 || len(value) > 64 {
		return false
	}
	for _, r := range value {
		isAlpha := (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
		isDigit := r >= '0' && r <= '9'
		if !isAlpha && !isDigit && r != '_' && r != '-' {
			return false
		}
	}
	return true
}

func isSchemaDigest(value string) bool {
	if len(value) != len("sha256:")+64 || !strings.HasPrefix(value, "sha256:") || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:"))
	return err == nil
}

func lowerBrokeredText(value string) string {
	return strings.ToLowerSpecial(brokeredUnicode15Case, value)
}

func hasUnsafeBrokeredText(value string) bool {
	lowered := lowerBrokeredText(value)
	return hasUnsafeBrokeredDescription(value) || containsBrokeredWord(lowered, "basic") || strings.Contains(lowered, brokeredTokenWord)
}

func hasUnsafeBrokeredDescription(value string) bool {
	lowered := lowerBrokeredText(value)
	normalized := normalizeKey(lowered)
	return containsSecretPrefix(value) || strings.Contains(value, "://") || containsBrokeredWord(lowered, "bearer") || containsBrokeredWord(lowered, brokeredSensitiveWord) || containsBrokeredWord(lowered, brokeredSensitivePluralWord) || containsBrokeredBasicAuthReference(value) || containsBrokeredCredentialAssignment(value) || containsBrokeredCredentialReference(value) || strings.Contains(lowered, authorizationKey) || strings.Contains(lowered, "secret") || strings.Contains(lowered, "password") || strings.Contains(lowered, "passphrase") || strings.Contains(lowered, "pwd") || strings.Contains(lowered, "api key") || strings.Contains(lowered, "apikey") || strings.Contains(normalized, "apikey") || strings.Contains(normalized, "xapikey") || strings.Contains(normalized, "subscriptionkey") || strings.Contains(normalized, "xfunctionskey") || strings.Contains(lowered, brokeredUnsafeCookieKey) || strings.Contains(lowered, "set-cookie") || strings.Contains(lowered, "x-api-key") || strings.Contains(lowered, credentialHeaderAPIKey) || strings.Contains(lowered, "subscription-key") || strings.Contains(lowered, "x-functions-key") || strings.Contains(lowered, "ocp-apim-subscription-key") || strings.Contains(lowered, "private key") || strings.Contains(lowered, "privatekey") || strings.Contains(lowered, "key material") || strings.Contains(lowered, ".svc") || strings.Contains(lowered, "cluster.local")
}

func containsBrokeredBasicAuthReference(value string) bool {
	lowered := lowerBrokeredText(value)
	if !containsBrokeredWord(lowered, "basic") {
		return false
	}
	if containsBrokeredWord(lowered, "auth") || containsBrokeredWord(lowered, brokeredAuthenticationWord) || containsBrokeredWord(lowered, authorizationKey) {
		return true
	}

	for _, match := range brokeredBasicValuePattern.FindAllStringSubmatch(value, -1) {
		if isBrokeredBasicValue(match[1]) {
			return true
		}
	}
	fields := strings.Fields(value)
	for _, field := range fields {
		if isBrokeredBasicValue(field) {
			return true
		}
	}
	for i, field := range fields {
		if !containsBrokeredWord(lowerBrokeredText(field), "basic") {
			continue
		}
		for j, candidate := range fields[i+1:] {
			if isBrokeredBasicValue(candidate) {
				return true
			}
			plain := strings.Trim(candidate, "\"'`()[]{}<>,;.")
			if strings.HasSuffix(plain, ":") && j+1 < len(fields[i+1:]) {
				return true
			}
			if strings.ContainsAny(candidate, ".!?;") {
				break
			}
		}
	}
	return false
}

func isBrokeredBasicValue(value string) bool {
	plain := strings.Trim(value, "\"'`()[]{}<>,;.")
	if separator := strings.IndexByte(plain, ':'); separator > 0 && separator+1 < len(plain) {
		return true
	}
	if decodesBrokeredBasicValue(value) {
		return true
	}
	for separator, r := range value {
		if (r == ':' || r == '=') && decodesBrokeredBasicValue(value[separator+1:]) {
			return true
		}
	}
	return false
}

func decodesBrokeredBasicValue(value string) bool {
	value = strings.TrimFunc(value, func(r rune) bool { return !isBrokeredBase64Rune(r) })
	if value == "" {
		return false
	}
	decoded, err := base64.StdEncoding.DecodeString(value)
	if err != nil {
		decoded, err = base64.RawStdEncoding.DecodeString(value)
	}
	return err == nil && bytes.Contains(decoded, []byte(":"))
}

func containsBrokeredCredentialAssignment(value string) bool {
	for separator := 0; separator < len(value); separator++ {
		if value[separator] != ':' && value[separator] != '=' {
			continue
		}
		if isHarmlessBrokeredTokenCountAssignment(value, separator) {
			continue
		}
		fields := strings.Fields(value[:separator])
		if len(fields) == 0 {
			continue
		}
		if isUnsafeBrokeredKey(fields[len(fields)-1]) {
			return true
		}
	}
	return false
}

func containsBrokeredCredentialReference(value string) bool {
	fields := strings.Fields(value)
	previous := ""
	for i, field := range fields {
		normalized := normalizeKey(field)
		if normalized == "" {
			continue
		}
		if normalized == brokeredTokenWord || normalized == brokeredTokensWord {
			if !isHarmlessBrokeredTokenUse(fields, i, normalizeKey(previous)) {
				return true
			}
			previous = field
			continue
		}
		if strings.Contains(normalized, brokeredTokenWord) && !isHarmlessBrokeredTokenWord(normalized) {
			return true
		}
		if hasBrokeredStructuredKeyShape(field) && isUnsafeBrokeredKey(field) {
			return true
		}
		switch normalized {
		case "accesstoken", "apitoken", "authtoken", "authenticationtoken", "authorizationtoken", "bearertoken", "credentialtoken", "identitytoken", "oauthtoken", "refreshtoken", "secrettoken", "sessiontoken":
			return true
		}
		previous = field
	}
	return false
}

func isHarmlessBrokeredTokenWord(value string) bool {
	switch value {
	case "tokenization", "tokenize", "tokenized", "tokenizer", "tokenizers", "tokenizing":
		return true
	default:
		return false
	}
}

func isHarmlessBrokeredTokenUse(fields []string, index int, previous string) bool {
	if hasBrokeredStructuredKeyShape(fields[index]) || !isHarmlessBrokeredTokenQualifier(previous) || !hasAdjacentBrokeredTokenCountIntent(fields, index) {
		return false
	}
	if index+1 == len(fields) {
		return true
	}
	return index+2 == len(fields) && isBrokeredNumericCount(fields[index+1])
}

func isHarmlessBrokeredTokenQualifier(value string) bool {
	switch value {
	case "completion", "context", "count", "counting", "counts", "input", "model", "output", "prompt", "usage":
		return true
	default:
		return false
	}
}

func hasAdjacentBrokeredTokenCountIntent(fields []string, tokenIndex int) bool {
	if tokenIndex > 0 && !endsBrokeredSentence(fields[tokenIndex-1]) && isHarmlessBrokeredTokenCountIntent(normalizeKey(fields[tokenIndex-1])) {
		return true
	}
	return tokenIndex > 1 && !endsBrokeredSentence(fields[tokenIndex-2]) && isHarmlessBrokeredTokenCountIntent(normalizeKey(fields[tokenIndex-2]))
}

func endsBrokeredSentence(value string) bool {
	value = strings.TrimRight(value, "\"'`)]}>,")
	return value != "" && strings.ContainsRune(".!?;", rune(value[len(value)-1]))
}

func isHarmlessBrokeredTokenCountIntent(value string) bool {
	switch value {
	case "count", "counting", "counts", "measure", "measures", "measuring", "report", "reporting", "reports", "track", "tracking":
		return true
	default:
		return false
	}
}

func isHarmlessBrokeredTokenCountAssignment(value string, separator int) bool {
	if value[separator] != ':' {
		return false
	}
	leftFields := strings.Fields(value[:separator])
	rightFields := strings.Fields(value[separator+1:])
	if len(leftFields) < 2 || len(rightFields) != 1 {
		return false
	}
	key := normalizeKey(leftFields[len(leftFields)-1])
	if key != brokeredTokenWord && key != brokeredTokensWord {
		return false
	}
	qualifier := normalizeKey(leftFields[len(leftFields)-2])
	return isHarmlessBrokeredTokenQualifier(qualifier) && hasAdjacentBrokeredTokenCountIntent(leftFields, len(leftFields)-1) && isBrokeredNumericCount(rightFields[0])
}

func isBrokeredNumericCount(value string) bool {
	wrapped := strings.TrimLeft(value, "([{<")
	if wrapped != "" && strings.ContainsRune("\"'`", rune(wrapped[0])) {
		return false
	}
	value = strings.Trim(value, "\"'`()[]{}<>,;:.")
	if value == "" {
		return false
	}
	parts := strings.Split(value, ",")
	digits := 0
	for i, part := range parts {
		if part == "" || i == 0 && len(parts) > 1 && len(part) > 3 || i > 0 && len(part) != 3 {
			return false
		}
		for _, r := range part {
			if r < '0' || r > '9' {
				return false
			}
			digits++
			if digits > 20 {
				return false
			}
		}
	}
	return digits > 0
}

func hasBrokeredStructuredKeyShape(value string) bool {
	return strings.HasPrefix(strings.TrimLeft(value, "\"'`([{<"), "-") || strings.ContainsAny(value, "_/.[]{}()<>\"'`") || value != lowerBrokeredText(value)
}

func containsBrokeredWord(value string, word string) bool {
	start := 0
	for {
		idx := strings.Index(value[start:], word)
		if idx < 0 {
			return false
		}
		idx += start
		after := idx + len(word)
		beforeOK := idx == 0 || !isBrokeredWordByte(value[idx-1])
		afterOK := after == len(value) || !isBrokeredWordByte(value[after])
		if beforeOK && afterOK {
			return true
		}
		start = after
	}
}

func isBrokeredWordByte(value byte) bool {
	return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') || (value >= '0' && value <= '9') || value == '_'
}

func isBrokeredBase64Rune(value rune) bool {
	return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') || (value >= '0' && value <= '9') || value == '+' || value == '/' || value == '='
}

func isUnsafeBrokeredKey(value string) bool {
	normalized := normalizeKey(value)
	switch normalized {
	case "auth", authorizationKey, "apikey", "bearer", brokeredUnsafeCookieKey, brokeredSensitiveWord, brokeredSensitivePluralWord, "endpoint", "endpoints", "executionendpoint", "executionurl", "header", "headers", "ocpapimsubscriptionkey", "password", "proxyauthorization", "secret", "secretref", "setcookie", "subscriptionkey", brokeredTokenWord, brokeredTokensWord, "url", "urls", "xapikey", "xfunctionskey":
		return true
	}
	authLike := (strings.HasPrefix(normalized, "auth") && !strings.HasPrefix(normalized, "author")) || strings.HasSuffix(normalized, "auth")
	return authLike || strings.Contains(normalized, authorizationKey) || strings.Contains(normalized, "header") || strings.Contains(normalized, "url") || strings.Contains(normalized, "endpoint") || strings.Contains(normalized, brokeredUnsafeCookieKey) || strings.Contains(normalized, "secret") || strings.Contains(normalized, brokeredTokenWord) || strings.Contains(normalized, "password") || strings.Contains(normalized, "passphrase") || strings.Contains(normalized, "pwd") || strings.Contains(normalized, "apikey") || strings.Contains(normalized, "accesskey") || strings.Contains(normalized, "privatekey") || strings.Contains(normalized, "keymaterial") || strings.Contains(normalized, brokeredSensitiveWord) || strings.Contains(normalized, "executionurl") || strings.Contains(normalized, "executionendpoint")
}

func normalizeKey(value string) string {
	var b strings.Builder
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r + ('a' - 'A'))
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		}
	}
	return b.String()
}

func canonicalNumber(value float64, bitSize int) ([]byte, error) {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return nil, fmt.Errorf("JSON numbers must be finite")
	}
	return []byte(strconv.FormatFloat(value, 'f', -1, bitSize)), nil
}

func canonicalJSONString(value string) ([]byte, error) {
	var encoded bytes.Buffer
	encoder := json.NewEncoder(&encoded)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return unescapeJSONLineSeparators(bytes.TrimSuffix(encoded.Bytes(), []byte("\n"))), nil
}

func unescapeJSONLineSeparators(encoded []byte) []byte {
	out := make([]byte, 0, len(encoded))
	for i := 0; i < len(encoded); {
		if encoded[i] != '\\' {
			out = append(out, encoded[i])
			i++
			continue
		}
		start := i
		for i < len(encoded) && encoded[i] == '\\' {
			i++
		}
		runLength := i - start
		if runLength%2 == 1 && i+5 <= len(encoded) && encoded[i] == 'u' {
			escape := string(encoded[i : i+5])
			if escape == "u2028" || escape == "u2029" {
				out = append(out, encoded[start:i-1]...)
				if escape == "u2028" {
					out = append(out, []byte("\u2028")...)
				} else {
					out = append(out, []byte("\u2029")...)
				}
				i += 5
				continue
			}
		}
		out = append(out, encoded[start:i]...)
	}
	return out
}

func canonicalJSONNumberString(value string) (string, error) {
	if !json.Valid([]byte(value)) {
		return "", fmt.Errorf("invalid JSON number %q", value)
	}
	number, ok := new(big.Rat).SetString(value)
	if !ok {
		return "", fmt.Errorf("invalid JSON number %q", value)
	}
	if number.IsInt() {
		if number.Sign() == 0 && strings.HasPrefix(value, "-") {
			return "-0", nil
		}
		return number.Num().String(), nil
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return "", err
	}
	if math.IsNaN(parsed) || math.IsInf(parsed, 0) {
		return "", fmt.Errorf("JSON numbers must be finite")
	}
	roundTrip, ok := new(big.Rat).SetString(strconv.FormatFloat(parsed, 'g', -1, 64))
	if !ok || roundTrip.Cmp(number) != 0 {
		return "", fmt.Errorf("JSON number %q cannot be represented exactly", value)
	}
	return strconv.FormatFloat(parsed, 'f', -1, 64), nil
}

func canonicalJSON(value any) ([]byte, error) {
	switch typed := value.(type) {
	case nil:
		return []byte("null"), nil
	case bool:
		if typed {
			return []byte("true"), nil
		}
		return []byte("false"), nil
	case string:
		return canonicalJSONString(typed)
	case int:
		return []byte(strconv.FormatInt(int64(typed), 10)), nil
	case int8:
		return []byte(strconv.FormatInt(int64(typed), 10)), nil
	case int16:
		return []byte(strconv.FormatInt(int64(typed), 10)), nil
	case int32:
		return []byte(strconv.FormatInt(int64(typed), 10)), nil
	case int64:
		return []byte(strconv.FormatInt(typed, 10)), nil
	case uint:
		return []byte(strconv.FormatUint(uint64(typed), 10)), nil
	case uint8:
		return []byte(strconv.FormatUint(uint64(typed), 10)), nil
	case uint16:
		return []byte(strconv.FormatUint(uint64(typed), 10)), nil
	case uint32:
		return []byte(strconv.FormatUint(uint64(typed), 10)), nil
	case uint64:
		return []byte(strconv.FormatUint(typed, 10)), nil
	case float32:
		return canonicalNumber(float64(typed), 32)
	case float64:
		return canonicalNumber(typed, 64)
	case json.Number:
		formatted, err := canonicalJSONNumberString(typed.String())
		if err != nil {
			return nil, err
		}
		return []byte(formatted), nil
	case []any:
		return canonicalJSONArray(typed)
	case []string:
		items := make([]any, len(typed))
		for i, item := range typed {
			items[i] = item
		}
		return canonicalJSONArray(items)
	case []int:
		items := make([]any, len(typed))
		for i, item := range typed {
			items[i] = item
		}
		return canonicalJSONArray(items)
	case []float64:
		items := make([]any, len(typed))
		for i, item := range typed {
			items[i] = item
		}
		return canonicalJSONArray(items)
	case []bool:
		items := make([]any, len(typed))
		for i, item := range typed {
			items[i] = item
		}
		return canonicalJSONArray(items)
	case map[string]any:
		return canonicalJSONObject(typed)
	default:
		return nil, fmt.Errorf("unsupported JSON value %T", value)
	}
}

func canonicalJSONArray(items []any) ([]byte, error) {
	var out bytes.Buffer
	out.WriteByte('[')
	for i, item := range items {
		if i > 0 {
			out.WriteByte(',')
		}
		encoded, err := canonicalJSON(item)
		if err != nil {
			return nil, err
		}
		out.Write(encoded)
	}
	out.WriteByte(']')
	return out.Bytes(), nil
}

func canonicalJSONObject(object map[string]any) ([]byte, error) {
	keys := make([]string, 0, len(object))
	for key := range object {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var out bytes.Buffer
	out.WriteByte('{')
	for i, key := range keys {
		if i > 0 {
			out.WriteByte(',')
		}
		encodedKey, err := canonicalJSONString(key)
		if err != nil {
			return nil, err
		}
		encodedValue, err := canonicalJSON(object[key])
		if err != nil {
			return nil, err
		}
		out.Write(encodedKey)
		out.WriteByte(':')
		out.Write(encodedValue)
	}
	out.WriteByte('}')
	return out.Bytes(), nil
}

// BrokeredToolSchemaDigest digests the exact safe schema surface AgentKit exposes to a model.
func BrokeredToolSchemaDigest(tool BrokeredTool) (string, error) {
	payload := map[string]any{
		"name":                       tool.Name,
		brokeredDigestDescriptionKey: tool.Description,
		"brokeredClass":              tool.BrokeredClass,
		"parameters":                 tool.Parameters,
	}
	encodedPayload, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	var normalizedPayload map[string]any
	decoder := json.NewDecoder(bytes.NewReader(encodedPayload))
	decoder.UseNumber()
	if err := decoder.Decode(&normalizedPayload); err != nil {
		return "", err
	}
	canonical, err := canonicalJSON(normalizedPayload)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(canonical)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func validateContext(add func(string, ...any), ctx Context, tools []Tool) {
	seen := map[string]bool{}
	toolByName := map[string]Tool{}
	for _, tool := range tools {
		if tool.Name != "" {
			toolByName[tool.Name] = tool
		}
	}
	for i, provider := range ctx.Providers {
		path := fmt.Sprintf("context.providers[%d]", i)
		if provider.Name != "" {
			if seen[provider.Name] {
				add("%s: duplicate provider name %q", path, provider.Name)
			}
			seen[provider.Name] = true
		}
		switch provider.Type {
		case ContextTypeSearch:
			validateEnvField(add, path+".endpointEnv", provider.EndpointEnv, true)
			validateEnvField(add, path+".indexEnv", provider.IndexEnv, true)
			validateContextAuth(add, path+".auth", provider.Auth)
		case ContextTypeSkills:
			if provider.Auth != nil {
				add("%s.auth must not be set for skills context providers; configure auth on the referenced MCP tool", path)
			}
			switch provider.Source {
			case ContextSourceFilesystem:
				if provider.Path == "" {
					add("%s.path is required for filesystem skills", path)
				} else if !isAgentSkillsPath(provider.Path) {
					add("%s.path %q must be an absolute path under /agent/skills; AgentKit does not copy arbitrary filesystem skill directories into images", path, provider.Path)
				}
			case ContextSourceMCP:
				if provider.ToolRef == "" {
					add("%s.toolRef is required for MCP skills", path)
				} else if tool, ok := toolByName[provider.ToolRef]; !ok {
					add("%s.toolRef %q references unknown tool", path, provider.ToolRef)
				} else if tool.URLEnv == "" || tool.Transport != ToolTransportStreamableHTTP {
					add("%s.toolRef %q must reference a streamable-http MCP tool", path, provider.ToolRef)
				}
			default:
				add("%s.source %q is not supported for skills (expected filesystem or mcp)", path, provider.Source)
			}
		case ContextTypeMemory:
			validateEnvField(add, path+".storeNameEnv", provider.StoreNameEnv, true)
			validateEnvField(add, path+".endpointEnv", provider.EndpointEnv, true)
			validateContextAuth(add, path+".auth", provider.Auth)
		case "":
			add("%s.type is required", path)
		default:
			add("%s.type %q is not supported (expected search, skills, or memory)", path, provider.Type)
		}
	}
}

func isAgentSkillsPath(path string) bool {
	clean := pathpkg.Clean(path)
	return clean == "/agent/skills" || strings.HasPrefix(clean, "/agent/skills/")
}

func validateContextAuth(add func(string, ...any), path string, auth *Auth) {
	validateAuth(add, path, auth)
	if auth != nil && auth.Type == AuthTypeBearer {
		add("%s.type %q is not supported for context providers; use %q", path, auth.Type, AuthTypeWorkloadIdentity)
	}
}

func validateObservability(add func(string, ...any), obs Observability) {
	validateEnvField(add, "observability.otel.endpointEnv", obs.OTel.EndpointEnv, false)
	validateEnvField(add, "observability.logs.levelEnv", obs.Logs.LevelEnv, false)
	if obs.Logs.LevelEnv != "" {
		add("observability.logs.levelEnv is not supported by current runtimes; omit it until log-level wiring is implemented")
	}
}

func validateEnvField(add func(string, ...any), path, value string, required bool) {
	if value == "" {
		if required {
			add("%s is required", path)
		}
		return
	}
	if !isEnvVarName(value) || looksLikeSecretLiteral(value) {
		add("%s %q must be an env var NAME matching [A-Z0-9_]+", path, value)
	}
}

func validateToolTransport(add func(string, ...any), i int, t Tool) {
	switch {
	case len(t.Command) > 0:
		if t.Transport != "" && t.Transport != ToolTransportStdio {
			add("tools[%d] (%s): command tools use transport %q or omit transport", i, t.Name, ToolTransportStdio)
		}
		if t.URLEnv != "" || len(t.Headers) > 0 || t.Auth != nil {
			add("tools[%d] (%s): stdio command tools must not set urlEnv, headers, or auth", i, t.Name)
		}
	case t.URLEnv != "":
		if t.Type != ToolTypeMCP {
			add("tools[%d] (%s): remote MCP tools must set type: %s", i, t.Name, ToolTypeMCP)
		}
		if t.Transport != ToolTransportStreamableHTTP {
			add("tools[%d] (%s): remote MCP tools must set transport: %s", i, t.Name, ToolTransportStreamableHTTP)
		}
		if !isEnvVarName(t.URLEnv) || looksLikeSecretLiteral(t.URLEnv) {
			add("tools[%d] (%s): urlEnv %q must be an env var NAME matching [A-Z0-9_]+", i, t.Name, t.URLEnv)
		}
		if len(t.Env) > 0 {
			add("tools[%d] (%s): remote MCP tools must use headers/auth instead of stdio env", i, t.Name)
		}
	case t.Transport != "" && t.Transport != ToolTransportStdio:
		add("tools[%d] (%s): transport %q requires urlEnv", i, t.Name, t.Transport)
	}
}

func validateToolEnv(add func(string, ...any), i int, t Tool) {
	for j, part := range t.Command {
		if part == "" {
			add("tools[%d] (%s): command[%d] must be non-empty", i, t.Name, j)
		}
	}
	for _, e := range t.Env {
		if e == "" {
			add("tools[%d] (%s): env entry is empty; list env var NAMES only", i, t.Name)
		} else if !isEnvVarName(e) || looksLikeSecretLiteral(e) {
			add("tools[%d] (%s): env entry %q must be an env var NAME matching [A-Z0-9_]+; list names only, never values", i, t.Name, e)
		}
	}
}

func validateToolHeaders(add func(string, ...any), i int, t Tool) {
	seen := map[string]bool{}
	for j, h := range t.Headers {
		path := fmt.Sprintf("tools[%d] (%s).headers[%d]", i, t.Name, j)
		switch {
		case h.Name == "":
			add("%s.name is required", path)
		case !isHTTPHeaderName(h.Name):
			add("%s.name %q is not a valid HTTP header name", path, h.Name)
		case seen[strings.ToLower(h.Name)]:
			add("%s: duplicate header name %q", path, h.Name)
		}
		seen[strings.ToLower(h.Name)] = true

		values := 0
		if h.Value != "" {
			values++
		}
		if h.ValueEnv != "" {
			values++
		}
		if values != 1 {
			add("%s must set exactly one of value or valueEnv", path)
		}
		if t.Auth != nil && strings.EqualFold(h.Name, authorizationKey) {
			add("%s must not set Authorization when auth is also configured; use one auth path", path)
		}
		if h.Value != "" && isCredentialHeaderName(h.Name) {
			add("%s.value must not bake a static credential header; use valueEnv or auth", path)
		}
		if h.Value != "" && hasSecretPrefix(h.Value) {
			add("%s.value looks like a secret value; use valueEnv instead", path)
		}
		if h.ValueEnv != "" && (!isEnvVarName(h.ValueEnv) || looksLikeSecretLiteral(h.ValueEnv)) {
			add("%s.valueEnv %q must be an env var NAME matching [A-Z0-9_]+", path, h.ValueEnv)
		}
	}
}

func validateAuth(add func(string, ...any), path string, auth *Auth) {
	if auth == nil {
		return
	}
	switch auth.Type {
	case AuthTypeBearer:
		if auth.TokenEnv == "" {
			add("%s.tokenEnv is required for bearer auth", path)
		} else if !isEnvVarName(auth.TokenEnv) || looksLikeSecretLiteral(auth.TokenEnv) {
			add("%s.tokenEnv %q must be an env var NAME matching [A-Z0-9_]+", path, auth.TokenEnv)
		}
		if auth.Audience != "" {
			add("%s.audience must be empty for bearer auth", path)
		}
	case AuthTypeWorkloadIdentity:
		if auth.Audience == "" {
			add("%s.audience is required for workload identity token auth", path)
		}
		if auth.TokenEnv != "" {
			add("%s.tokenEnv must be empty for workload identity token auth", path)
		}
	default:
		add("%s.type %q is not supported (expected %q or %q)", path, auth.Type, AuthTypeBearer, AuthTypeWorkloadIdentity)
	}
}

func validateApproval(add func(string, ...any), i int, t Tool) {
	switch t.Approval {
	case "", ApprovalNever, ApprovalAuto, ApprovalAlways:
	default:
		add("tools[%d] (%s): approval %q is not supported (expected never, auto, or always)", i, t.Name, t.Approval)
	}
}

func (c *AgentConfig) requiredCapabilities() []string {
	seen := map[string]bool{}
	var out []string
	add := func(capability string) {
		if !seen[capability] {
			seen[capability] = true
			out = append(out, capability)
		}
	}
	if c.Model.Auth != nil && c.Model.Auth.Type == AuthTypeWorkloadIdentity {
		add(runtimes.CapabilityModelWorkloadIdentityAuth)
	}
	for _, t := range c.Tools {
		if len(t.Command) > 0 {
			add(runtimes.CapabilityStdioMCP)
		}
		if t.URLEnv != "" || t.Transport == ToolTransportStreamableHTTP {
			add(runtimes.CapabilityStreamableHTTPMCP)
		}
		if t.Auth != nil && t.Auth.Type == AuthTypeWorkloadIdentity {
			add(runtimes.CapabilityWorkloadIdentityTokenAuth)
		}
		if t.Approval == ApprovalAuto || t.Approval == ApprovalAlways {
			add(runtimes.CapabilityToolApproval)
		}
	}
	for _, provider := range c.Context.Providers {
		switch provider.Type {
		case ContextTypeSearch:
			add(runtimes.CapabilityContextProviderSearch)
		case ContextTypeSkills:
			add(runtimes.CapabilityContextProviderSkills)
			if provider.Source == ContextSourceFilesystem {
				add(runtimes.CapabilityFilesystemSkills)
			}
			if provider.Source == ContextSourceMCP {
				add(runtimes.CapabilityMCPSkills)
			}
		case ContextTypeMemory:
			add(runtimes.CapabilityContextProviderMemory)
		}
		if provider.Auth != nil && provider.Auth.Type == AuthTypeWorkloadIdentity {
			add(runtimes.CapabilityWorkloadIdentityTokenAuth)
		}
	}
	if c.Observability.OTel.EndpointEnv != "" {
		add(runtimes.CapabilityOTelExport)
	}
	return out
}

func isEnvVarName(v string) bool {
	if v == "" {
		return false
	}
	for _, r := range v {
		isUpper := r >= 'A' && r <= 'Z'
		isDigit := r >= '0' && r <= '9'
		if !isUpper && !isDigit && r != '_' {
			return false
		}
	}
	return true
}

func isHTTPHeaderName(v string) bool {
	if v == "" {
		return false
	}
	for _, r := range v {
		isAlpha := (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
		isDigit := r >= '0' && r <= '9'
		switch {
		case isAlpha || isDigit:
			continue
		case strings.ContainsRune("!#$%&'*+-.^_`|~", r):
			continue
		default:
			return false
		}
	}
	return true
}

func isCredentialHeaderName(name string) bool {
	switch strings.ToLower(name) {
	case authorizationKey, "proxy-authorization", brokeredUnsafeCookieKey, "set-cookie", "x-api-key", credentialHeaderAPIKey, "ocp-apim-subscription-key", "subscription-key", "x-functions-key":
		return true
	default:
		return false
	}
}

func hasSecretPrefix(v string) bool {
	for _, p := range []string{"sk-", "sk_", "ghp_", "github_pat_", "xoxb-", "AKIA"} {
		if strings.HasPrefix(v, p) {
			return true
		}
	}
	return false
}

func containsSecretPrefix(v string) bool {
	for _, p := range []string{"sk-", "sk_", "ghp_", "github_pat_", "xoxb-", "AKIA"} {
		if strings.Contains(v, p) {
			return true
		}
	}
	return false
}

// looksLikeSecretLiteral heuristically flags a value that appears to be a secret
// rather than an env var NAME. Env var names are uppercase letters, digits, and
// underscores; common secret prefixes (sk-, etc.) and lowercase/punctuation are
// strong signals the user pasted a value where a name belongs.
func looksLikeSecretLiteral(v string) bool {
	if v == "" {
		return false
	}
	// Known secret-value prefixes.
	if hasSecretPrefix(v) {
		return true
	}
	// Env var NAMEs are [A-Z0-9_]+; lowercase/spaces/URLs/punctuation are a strong
	// signal the user pasted a value into an env-name field.
	if isEnvVarName(v) {
		return false
	}
	for _, r := range v {
		isUpper := r >= 'A' && r <= 'Z'
		isDigit := r >= '0' && r <= '9'
		if !isUpper && !isDigit && r != '_' {
			return true
		}
	}
	return false
}
