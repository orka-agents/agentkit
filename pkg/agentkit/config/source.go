package config

import (
	"fmt"

	"github.com/goccy/go-yaml"
)

// Source is a tagged-union reference to authored content (plan §7 "sources as a
// tagged union"). v0 wires two variants:
//
//	inline — content authored directly in the file (bare string or {inline: ...})
//	file   — a path in the build context (e.g. {file: ./prompt.md})
//
// The struct-of-pointers shape models each variant as a distinct, optional
// field; ExactlyOne reports which is set. v1 variants (http, git, image) are
// added as new fields here, not as new switch arms scattered across call sites
// (plan §16.2 #2). Bare-string YAML (instructions: |) decodes into Inline.
type Source struct {
	// Inline is content authored directly in the agentkitfile.
	Inline string `yaml:"inline,omitempty"`
	// File is a path relative to the build context.
	File string `yaml:"file,omitempty"`
}

// sourceStruct mirrors Source for the mapping form, without the custom
// unmarshaler (avoids infinite recursion).
type sourceStruct struct {
	Inline string `yaml:"inline,omitempty"`
	File   string `yaml:"file,omitempty"`
}

// UnmarshalYAML implements goccy's BytesUnmarshaler so a Source accepts either a
// bare scalar (→ Inline) or a mapping with explicit variant keys. Both forms are
// decoded strictly so an unknown key (e.g. a misspelled `fil:`) is a load-time
// error rather than a silently-empty source.
func (s *Source) UnmarshalYAML(b []byte) error {
	// Try the scalar form first: `instructions: |  ...`.
	var scalar string
	if err := yaml.Unmarshal(b, &scalar); err == nil {
		s.Inline = scalar
		return nil
	}

	// Fall back to the mapping form: `instructions: { file: ./prompt.md }`.
	var aux sourceStruct
	if err := yaml.UnmarshalWithOptions(b, &aux, strictDecodeOptions()...); err != nil {
		return fmt.Errorf("instructions/source must be a string or a {inline|file} mapping: %w", err)
	}
	s.Inline = aux.Inline
	s.File = aux.File
	return nil
}

// IsZero reports whether no source variant is set.
func (s Source) IsZero() bool {
	return s.Inline == "" && s.File == ""
}

// variantsSet returns the names of the populated variants (for exactly-one checks).
func (s Source) variantsSet() []string {
	var set []string
	if s.Inline != "" {
		set = append(set, "inline")
	}
	if s.File != "" {
		set = append(set, "file")
	}
	return set
}
