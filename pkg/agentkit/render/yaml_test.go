package render

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestDecodeJSONDocumentsKeepsDocumentsAfterEmptyOnes(t *testing.T) {
	// goccy's parser alone drops every document after `---\n---\n`.
	docs, err := decodeJSONDocuments([]byte("---\n---\n# Source: chart/tools.yaml\n---\na: 1\n---\nnull\n---\nb: |\n  ---\n  text\n"))
	if err != nil {
		t.Fatalf("decodeJSONDocuments() error = %v", err)
	}
	want := []any{map[string]any{"a": json.Number("1")}, map[string]any{"b": "---\ntext\n"}}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs = %#v, want %#v", docs, want)
	}
}

func TestDecodeJSONDocumentsKeepsNumberText(t *testing.T) {
	docs, err := decodeJSONDocuments([]byte(`{"json": [1, 2.50, -0.0, 1.5E+3], "big": 99999999999999999999999}
---
half: .5
whole: 1.
leading: 01.5
hex: 0x1F
exact: 9007199254740993.0
quoted: "123"
tagged: !!str 123
`))
	if err != nil {
		t.Fatalf("decodeJSONDocuments() error = %v", err)
	}
	want := []any{
		map[string]any{
			"json": []any{json.Number("1"), json.Number("2.50"), json.Number("-0.0"), json.Number("1500")},
			"big":  json.Number("99999999999999999999999"),
		},
		map[string]any{
			"half":    json.Number("0.5"),
			"whole":   json.Number("1"),
			"leading": json.Number("1.5"),
			"hex":     json.Number("31"),
			"exact":   json.Number("9007199254740993"),
			"quoted":  "123",
			"tagged":  "123",
		},
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs = %#v, want %#v", docs, want)
	}
}

func TestDecodeJSONDocumentsDecodesCoreTagsByType(t *testing.T) {
	docs, err := decodeJSONDocuments([]byte(`int: !!int 3
quotedInt: !!int "42"
bigInt: !!int "9007199254740993"
float: !!float "0.5"
wholeFloat: !!float 3
bool: !!bool "true"
null: !!null ""
str: !!str 1.5
map: !!map {a: 1}
seq: !!seq [1]
`))
	if err != nil {
		t.Fatalf("decodeJSONDocuments() error = %v", err)
	}
	want := []any{map[string]any{
		"int":        json.Number("3"),
		"quotedInt":  json.Number("42"),
		"bigInt":     json.Number("9007199254740993"),
		"float":      json.Number("0.5"),
		"wholeFloat": json.Number("3"),
		"bool":       true,
		"null":       nil,
		"str":        "1.5",
		"map":        map[string]any{"a": json.Number("1")},
		"seq":        []any{json.Number("1")},
	}}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs = %#v, want %#v", docs, want)
	}
}

func TestDecodeJSONDocumentsResolvesAnchorsAndMerges(t *testing.T) {
	docs, err := decodeJSONDocuments([]byte(`base: &base {type: string, minLength: 1}
site:
  <<: *base
  minLength: 2
tags: [*base]
`))
	if err != nil {
		t.Fatalf("decodeJSONDocuments() error = %v", err)
	}
	doc, _ := docs[0].(map[string]any)
	site, _ := doc["site"].(map[string]any)
	if site[schemaTypeKey] != schemaTypeString || site["minLength"] != json.Number("2") {
		t.Fatalf("merged mapping = %#v", site)
	}
	if !reflect.DeepEqual(doc["tags"], []any{doc["base"]}) {
		t.Fatalf("alias = %#v", doc["tags"])
	}
}

func TestDecodeJSONDocumentsRejectsNonJSONValues(t *testing.T) {
	for _, tc := range []struct {
		source string
		want   string
	}{
		{"value: .inf\n", "must be finite"},
		{"value: !!binary aGk=\n", "YAML tag !!binary is not supported"},
		{"value: !!int 3.0\n", `YAML !!int value "3.0" is not an integer`},
		{"value: !!float .inf\n", "is not a valid JSON number"},
		{"value: !!float \"\"\n", "is not a valid JSON number"},
		{"value: *missing\n", `YAML alias "missing" has no anchor`},
		{"a: 1\na: 2\n", `mapping key "a" already defined`},
	} {
		if _, err := decodeJSONDocuments([]byte(tc.source)); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("decodeJSONDocuments(%q) error = %v, want %q", tc.source, err, tc.want)
		}
	}
	if _, err := floatNumber("1.0e99999"); err == nil || !strings.Contains(err.Error(), "out of range") {
		t.Fatalf("floatNumber() error = %v, want exponent bound", err)
	}
}
