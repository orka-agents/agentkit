package config

import (
	"errors"
	"fmt"

	"github.com/goccy/go-yaml"
	"github.com/orka-agents/agentkit/pkg/utils"
)

// probe is the minimal {apiVersion, kind} discriminator read before full
// unmarshal. Reading kind explicitly is the fix for AIKit's silent-misparse bug
// (plan §5.1): dispatch on kind, never on an implicit type collision.
type probe struct {
	APIVersion string `yaml:"apiVersion"`
	Kind       string `yaml:"kind"`
}

// Document is one parsed agentkitfile. Exactly one field is set, chosen by kind.
type Document struct {
	Agent   *AgentConfig
	Toolbox *ToolboxConfig
}

const supportedKinds = "`kind: Agent` or `kind: Toolbox`"

// NewDocumentFromBytes parses an agentkitfile of any supported kind. Like
// NewFromBytes, it dispatches on an explicit kind and decodes strictly.
func NewDocumentFromBytes(b []byte) (*Document, error) {
	kind, err := probeKind(b)
	if err != nil {
		return nil, err
	}
	switch kind {
	case "":
		return nil, errors.New("kind is not defined; expected " + supportedKinds)
	case utils.KindAgent:
		cfg := &AgentConfig{}
		if err := decodeStrict(b, cfg); err != nil {
			return nil, err
		}
		return &Document{Agent: cfg}, nil
	case utils.KindToolbox:
		cfg := &ToolboxConfig{}
		if err := decodeStrict(b, cfg); err != nil {
			return nil, err
		}
		return &Document{Toolbox: cfg}, nil
	default:
		return nil, fmt.Errorf("kind %q is not supported; expected %s", kind, supportedKinds)
	}
}

// NewFromBytes parses an agentkitfile. It first reads a {apiVersion, kind}
// probe (non-strict, so the probe tolerates other fields), dispatches on kind,
// then strict-unmarshals the full AgentConfig so unknown/misspelled fields are
// load-time errors with line:column positions.
//
// Only kind: Agent is supported. A kind-less file is rejected with a clear
// message rather than silently misparsing.
func NewFromBytes(b []byte) (*AgentConfig, error) {
	kind, err := probeKind(b)
	if err != nil {
		return nil, err
	}
	if kind == "" {
		return nil, errors.New("kind is not defined; expected `kind: Agent`")
	}
	if kind != utils.KindAgent {
		return nil, fmt.Errorf("kind %q is not supported; expected `kind: Agent`", kind)
	}

	cfg := &AgentConfig{}
	if err := decodeStrict(b, cfg); err != nil {
		return nil, err
	}
	return cfg, nil
}

func probeKind(b []byte) (string, error) {
	var p probe
	if err := yaml.Unmarshal(b, &p); err != nil {
		return "", fmt.Errorf("reading apiVersion/kind probe: %w", formatYAMLError(b, err))
	}
	return p.Kind, nil
}

func decodeStrict(b []byte, out any) error {
	if err := yaml.UnmarshalWithOptions(b, out, yaml.Strict()); err != nil {
		return fmt.Errorf("parsing agentkitfile: %w", formatYAMLError(b, err))
	}
	return nil
}

// formatYAMLError enriches a goccy decode error with line:column context.
func formatYAMLError(_ []byte, err error) error {
	if err == nil {
		return nil
	}
	// FormatError renders the offending line and a caret; colored=false for logs.
	return errors.New(yaml.FormatError(err, false, true))
}
