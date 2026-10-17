package render

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"sort"
	"strings"

	"github.com/orka-agents/agentkit/pkg/agentkit/abi"
	"github.com/orka-agents/agentkit/pkg/agentkit/config"
)

const (
	TargetBrokeredTools = "brokered-tools"
	orkaToolAPIVersion  = "core.orka.ai/v1alpha1"
	orkaToolKind        = "Tool"
	schemaTypeKey       = "type"
	schemaTypeObject    = "object"
	schemaTypeString    = "string"
)

var orkaBrokeredToolClasses = []string{config.BrokeredClassRead, config.BrokeredClassWrite, config.BrokeredClassCoordination}

// BrokeredToolsFromFiles renders the safe, model-facing subset of the Orka Tool
// CRDs in the given YAML/JSON files as an agent.yaml brokeredTools fragment.
// Only name, description, brokered class, parameters schema, and an optional
// schema digest are exported; execution URLs, headers, and Secret references
// never leave the CRD. With bare set, only the list is emitted.
func BrokeredToolsFromFiles(paths []string, includeDigest, bare bool) ([]byte, error) {
	if len(paths) == 0 {
		return nil, errors.New("at least one Orka Tool CRD file is required")
	}
	var documents []any
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		docs, err := decodeJSONDocuments(data)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		documents = append(documents, docs...)
	}
	tools, err := OrkaBrokeredTools(documents, includeDigest)
	if err != nil {
		return nil, err
	}
	return abi.RenderBrokeredTools(tools, bare)
}

// OrkaBrokeredTools converts decoded Orka Tool CRD documents into validated
// brokeredTools sorted by name. The CRD stays the source of truth: only the
// canonical core.orka.ai/v1alpha1 Tool shape is accepted, tools without
// spec.brokeredToolClass are not brokered and are skipped, and an input set
// with no brokered tools is rejected.
func OrkaBrokeredTools(documents []any, includeDigest bool) ([]config.BrokeredTool, error) {
	var tools []config.BrokeredTool
	for idx, document := range documents {
		tool, brokered, err := orkaBrokeredTool(idx, document)
		if err != nil {
			return nil, err
		}
		if !brokered {
			continue
		}
		if err := config.ValidateBrokeredTool(tool); err != nil {
			return nil, fmt.Errorf("tool CRD %q: %w", tool.Name, err)
		}
		if includeDigest {
			digest, err := config.BrokeredToolSchemaDigest(tool)
			if err != nil {
				return nil, fmt.Errorf("tool CRD %q: %w", tool.Name, err)
			}
			tool.SchemaDigest = digest
		}
		tools = append(tools, tool)
	}
	if len(tools) == 0 {
		return nil, errors.New("no brokered Orka Tool CRDs found; set spec.brokeredToolClass to read, write, or coordination")
	}
	sort.SliceStable(tools, func(i, j int) bool { return tools[i].Name < tools[j].Name })
	var duplicates []string
	for i := 1; i < len(tools); i++ {
		if name := tools[i].Name; name == tools[i-1].Name && !slices.Contains(duplicates, name) {
			duplicates = append(duplicates, name)
		}
	}
	if len(duplicates) > 0 {
		return nil, fmt.Errorf("duplicate brokered tool names: %s", strings.Join(duplicates, ", "))
	}
	return tools, nil
}

func orkaBrokeredTool(idx int, document any) (config.BrokeredTool, bool, error) {
	resource, ok := document.(map[string]any)
	if !ok {
		return config.BrokeredTool{}, false, fmt.Errorf("document %d: Orka Tool CRD must be an object", idx)
	}
	apiVersion, kind := resource["apiVersion"], resource["kind"]
	if apiVersion != orkaToolAPIVersion || kind != orkaToolKind {
		return config.BrokeredTool{}, false, fmt.Errorf("unsupported Orka Tool GVK in document %d: apiVersion=%v, kind=%v; expected apiVersion=%s, kind=%s", idx, apiVersion, kind, orkaToolAPIVersion, orkaToolKind)
	}
	metadata, ok := resource["metadata"].(map[string]any)
	if !ok {
		return config.BrokeredTool{}, false, fmt.Errorf("document %d: Orka Tool CRD metadata must be an object", idx)
	}
	spec, ok := resource["spec"].(map[string]any)
	if !ok {
		return config.BrokeredTool{}, false, fmt.Errorf("document %d: Orka Tool CRD spec must be an object", idx)
	}
	rawClass, brokered := spec["brokeredToolClass"]
	if !brokered {
		return config.BrokeredTool{}, false, nil
	}
	name, ok := metadata["name"].(string)
	if !ok {
		return config.BrokeredTool{}, false, fmt.Errorf("document %d: Orka Tool CRD is missing metadata.name", idx)
	}
	description, ok := spec["description"].(string)
	if !ok {
		return config.BrokeredTool{}, false, fmt.Errorf("tool CRD %q description must be a string", name)
	}
	brokeredClass, ok := rawClass.(string)
	if !ok {
		return config.BrokeredTool{}, false, fmt.Errorf("tool CRD %q spec.brokeredToolClass must be a string", name)
	}
	if !slices.Contains(orkaBrokeredToolClasses, brokeredClass) {
		return config.BrokeredTool{}, false, fmt.Errorf("tool CRD %q has unsupported spec.brokeredToolClass %q; expected %s", name, brokeredClass, strings.Join(orkaBrokeredToolClasses, ", "))
	}
	parameters := map[string]any{schemaTypeKey: schemaTypeObject}
	if raw, ok := spec["parameters"]; ok && raw != nil {
		if parameters, ok = raw.(map[string]any); !ok {
			return config.BrokeredTool{}, false, fmt.Errorf("tool CRD %q parameters schema must be an object", name)
		}
	}
	return config.BrokeredTool{
		Name:          name,
		Description:   description,
		BrokeredClass: brokeredClass,
		Parameters:    parameters,
	}, true, nil
}
