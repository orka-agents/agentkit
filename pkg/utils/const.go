// Package utils holds dependency-light constants shared across the AgentKit frontend.
package utils

const (
	// APIv1alpha1 is the only supported apiVersion for v0.
	APIv1alpha1 = "v1alpha1"

	// KindAgent is the config discriminator for an agent build (see §5.1).
	KindAgent = "Agent"

	// KindToolbox is the config discriminator for an Orka toolbox image build.
	KindToolbox = "Toolbox"

	// ProviderOpenAICompatible is the only model provider supported in v0.
	ProviderOpenAICompatible = "openai-compatible"

	// OutputKindImage is the default (and only v0) output kind.
	OutputKindImage = "image"

	// Platform constants.
	PlatformLinux = "linux"
	PlatformAMD64 = "amd64"
	PlatformARM64 = "arm64"

	// AgentKitRoot is where the runtime adapter (agentkit-serve) is staged.
	AgentKitRoot = "/opt/agentkit"

	// ServeBinary is the agentkit-serve entrypoint inside the adapter payload.
	ServeBinary = AgentKitRoot + "/bin/agentkit-serve"

	// DefaultPort is the default standalone/OpenAI serve port.
	DefaultPort = 8080

	// DefaultFoundryPort is the default Foundry protocol serve port when the
	// agent ABI kept the generic default port.
	DefaultFoundryPort = 8088

	// LabelPrefix namespaces all AgentKit OCI labels (io.github.<owner>.agentkit.*).
	LabelPrefix = "io.github.orka-agents.agentkit"
)
