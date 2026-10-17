package render

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"regexp"

	"github.com/goccy/go-yaml"
)

const (
	TargetFoundryHostedAgent = "foundry-hosted-agent"
	foundryWrapperImageKind  = "foundry-wrapper"
	foundryProtocolVersion   = "1.0.0"
)

var foundryEnvNamePattern = regexp.MustCompile(`^[A-Z0-9_]+$`)

// foundryProfile is the provider-neutral deployment profile. It stays outside
// AgentKit core; unknown keys (such as the source agentkitfile) are ignored.
type foundryProfile struct {
	Target    string `yaml:"target"`
	Name      string `yaml:"name"`
	AgentName string `yaml:"agentName"`
	Metadata  struct {
		Name string `yaml:"name"`
	} `yaml:"metadata"`
	Image          string `yaml:"image"`
	FoundryWrapper bool   `yaml:"foundryWrapper"`
	ImageKind      string `yaml:"imageKind"`
	Protocols      []any  `yaml:"protocols"`
	Resources      any    `yaml:"resources"`
	Env            any    `yaml:"env"`
	Environment    any    `yaml:"environment"`
}

type foundryHostedAgent struct {
	Kind      string            `yaml:"kind"`
	Name      string            `yaml:"name"`
	Image     string            `yaml:"image"`
	Protocols []foundryProtocol `yaml:"protocols"`
	Resources any               `yaml:"resources"`
	// Foundry/azd sample-compatible snake_case list. The camelCase map form can
	// validate locally yet fail to inject variables into hosted containers.
	EnvironmentVariables []foundryEnvVar `yaml:"environment_variables,omitempty"`
}

type foundryProtocol struct {
	Protocol string `yaml:"protocol"`
	Version  string `yaml:"version"`
}

type foundryEnvVar struct {
	Name  string `yaml:"name"`
	Value string `yaml:"value"`
}

// FoundryHostedAgentFromFile renders an azd/Foundry hosted ContainerAgent
// definition from a provider-neutral deployment profile file.
func FoundryHostedAgentFromFile(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	out, err := FoundryHostedAgent(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return out, nil
}

// FoundryHostedAgent maps a provider-neutral deployment profile onto Foundry's
// hosted-agent definition. The profile image must already serve Foundry's
// /readiness, /invocations, and /responses contract, so a plain AgentKit /v1
// image is never advertised as a Foundry-hosted protocol image.
func FoundryHostedAgent(profileYAML []byte) ([]byte, error) {
	var profile foundryProfile
	if err := yaml.UnmarshalWithOptions(profileYAML, &profile, yaml.UseOrderedMap()); err != nil {
		return nil, fmt.Errorf("parsing profile: %w", err)
	}
	if profile.Target != "" && profile.Target != TargetFoundryHostedAgent {
		return nil, fmt.Errorf("profile target must be %s", TargetFoundryHostedAgent)
	}
	name := firstNonEmpty(profile.Name, profile.AgentName, profile.Metadata.Name)
	if name == "" {
		return nil, errors.New("profile must set name")
	}
	if profile.Image == "" {
		return nil, errors.New("profile must set image")
	}
	if !profile.FoundryWrapper && profile.ImageKind != foundryWrapperImageKind {
		return nil, errors.New("profile image must be a Foundry protocol wrapper image; set foundryWrapper: true after wrapping the AgentKit image with foundry_live.py")
	}
	protocols, err := foundryProtocols(profile.Protocols)
	if err != nil {
		return nil, err
	}
	env := profile.Env
	if isEmptyYAMLValue(env) {
		env = profile.Environment
	}
	envVars, err := foundryEnvVars(env)
	if err != nil {
		return nil, err
	}
	resources := profile.Resources
	if isEmptyYAMLValue(resources) {
		resources = yaml.MapSlice{{Key: "cpu", Value: "0.25"}, {Key: "memory", Value: "0.5Gi"}}
	}
	out, err := yaml.Marshal(foundryHostedAgent{
		Kind:                 "hosted",
		Name:                 name,
		Image:                profile.Image,
		Protocols:            protocols,
		Resources:            resources,
		EnvironmentVariables: envVars,
	})
	if err != nil {
		return nil, err
	}
	return append(bytes.TrimSpace(out), '\n'), nil
}

func foundryProtocols(values []any) ([]foundryProtocol, error) {
	if values == nil {
		values = []any{"invocations", "responses"}
	}
	out := make([]foundryProtocol, 0, len(values))
	for _, item := range values {
		switch typed := item.(type) {
		case string:
			out = append(out, foundryProtocol{Protocol: typed, Version: foundryProtocolVersion})
		case yaml.MapSlice:
			protocol, ok := mapSliceValue(typed, "protocol")
			if !ok {
				return nil, fmt.Errorf("invalid protocol entry: %v", item)
			}
			version, ok := mapSliceValue(typed, "version")
			if !ok {
				version = foundryProtocolVersion
			}
			out = append(out, foundryProtocol{Protocol: fmt.Sprint(protocol), Version: fmt.Sprint(version)})
		default:
			return nil, fmt.Errorf("invalid protocol entry: %v", item)
		}
	}
	return out, nil
}

func foundryEnvVars(raw any) ([]foundryEnvVar, error) {
	type pair struct {
		name  string
		value any
	}
	var pairs []pair
	switch typed := raw.(type) {
	case nil:
	case yaml.MapSlice:
		for _, item := range typed {
			pairs = append(pairs, pair{name: fmt.Sprint(item.Key), value: item.Value})
		}
	case []any:
		for _, item := range typed {
			entry, ok := item.(yaml.MapSlice)
			name, hasName := mapSliceValue(entry, "name")
			if !ok || !hasName {
				return nil, fmt.Errorf("env list entries must be mappings with name/value: %v", item)
			}
			value, _ := mapSliceValue(entry, "value")
			pairs = append(pairs, pair{name: fmt.Sprint(name), value: value})
		}
	default:
		return nil, errors.New("env must be a mapping or list")
	}
	out := make([]foundryEnvVar, 0, len(pairs))
	for _, p := range pairs {
		if !foundryEnvNamePattern.MatchString(p.name) {
			return nil, fmt.Errorf("env name %q must match [A-Z0-9_]+", p.name)
		}
		value := "${" + p.name + "}"
		if p.value != nil {
			value = fmt.Sprint(p.value)
		}
		out = append(out, foundryEnvVar{Name: p.name, Value: value})
	}
	return out, nil
}

func mapSliceValue(m yaml.MapSlice, key string) (any, bool) {
	for _, item := range m {
		if item.Key == key {
			return item.Value, true
		}
	}
	return nil, false
}

func isEmptyYAMLValue(value any) bool {
	switch typed := value.(type) {
	case nil:
		return true
	case yaml.MapSlice:
		return len(typed) == 0
	case []any:
		return len(typed) == 0
	default:
		return false
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
