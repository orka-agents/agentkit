package config

import (
	"sort"
	"strings"

	"github.com/orka-agents/agentkit/pkg/utils"
)

const (
	nativeImageLabelNamespace   = utils.LabelPrefix
	portableImageLabelNamespace = "ai.agentkit"
	orkaImageLabelNamespace     = "ai.orka"
	// legacyNativeImageLabelNamespace is the pre-orka-agents native namespace.
	// It stays reserved so user labels cannot spoof it on newly built images.
	legacyNativeImageLabelNamespace = "ai.sozercan.agentkit"

	// ImageLabelNativeRuntime identifies the canonical AgentKit runtime.
	ImageLabelNativeRuntime = nativeImageLabelNamespace + ".runtime"
	// ImageLabelNativeName identifies the authored AgentKit agent name.
	ImageLabelNativeName = nativeImageLabelNamespace + ".name"
	// ImageLabelNativeABI identifies the baked Agent YAML ABI version.
	ImageLabelNativeABI = nativeImageLabelNamespace + ".abi"

	// ImageLabelPortableABI is the cross-orchestrator Agent YAML ABI label.
	ImageLabelPortableABI = portableImageLabelNamespace + ".abi"
	// ImageLabelPortableRuntime is the cross-orchestrator runtime identity label.
	ImageLabelPortableRuntime = portableImageLabelNamespace + ".runtime"
	// ImageLabelPortableProtocols lists the protocols exposed by AgentKit images.
	ImageLabelPortableProtocols = portableImageLabelNamespace + ".protocols"
	// ImageLabelPortableCapabilities lists the selected runtime's capabilities.
	ImageLabelPortableCapabilities = portableImageLabelNamespace + ".capabilities"

	// ImageLabelOrkaHarnessVersion identifies the supported Orka harness contract.
	ImageLabelOrkaHarnessVersion = orkaImageLabelNamespace + ".harness.version"
	// ImageLabelOrkaToolbox describes where a toolbox image keeps its tools.
	ImageLabelOrkaToolbox = orkaImageLabelNamespace + ".toolbox"

	// ImageLabelOCITitle is the standard OCI title generated from metadata.name.
	ImageLabelOCITitle = "org.opencontainers.image.title"
)

var reservedMetadataLabelNamespaces = [...]string{
	nativeImageLabelNamespace,
	legacyNativeImageLabelNamespace,
	portableImageLabelNamespace,
	orkaImageLabelNamespace,
}

var reservedMetadataLabelKeys = [...]string{
	ImageLabelOCITitle,
}

func reservedMetadataLabelNamespace(key string) (string, bool) {
	for _, namespace := range reservedMetadataLabelNamespaces {
		if key == namespace || strings.HasPrefix(key, namespace+".") {
			return namespace, true
		}
	}
	return "", false
}

func isReservedMetadataLabelKey(key string) bool {
	for _, reserved := range reservedMetadataLabelKeys {
		if key == reserved {
			return true
		}
	}
	return false
}

// validateMetadataLabels rejects user labels that would spoof AgentKit- or
// Orka-generated image metadata. Every document kind shares this check.
func validateMetadataLabels(add func(string, ...any), labels map[string]string) {
	keys := make([]string, 0, len(labels))
	for key := range labels {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		if namespace, reserved := reservedMetadataLabelNamespace(key); reserved {
			add("metadata.labels[%q] uses reserved AgentKit/Orka control-plane label namespace %q", key, namespace)
			continue
		}
		if isReservedMetadataLabelKey(key) {
			add("metadata.labels[%q] is reserved for AgentKit-generated image identity metadata", key)
		}
	}
}
