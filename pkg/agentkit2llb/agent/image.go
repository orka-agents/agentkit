package agent

import (
	"fmt"
	"strings"

	"github.com/moby/buildkit/util/system"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/orka-agents/agentkit/pkg/agentkit/abi"
	"github.com/orka-agents/agentkit/pkg/agentkit/config"
	"github.com/orka-agents/agentkit/pkg/agentkit/effective"
	"github.com/orka-agents/agentkit/pkg/agentkit/runtimes"
	"github.com/orka-agents/agentkit/pkg/utils"
)

const (
	// All shipped adapters select these modes at runtime. AgentSessions still
	// requires verified digests, a text-only profile and host-mediated execution.
	imageProtocols     = "openai,foundry,orka,agentsessions"
	orkaHarnessVersion = "orka.harness.v1"
)

// NewImageConfig builds the OCI image config for the agent image. It deliberately
// does NOT inherit AIKit's root user: per plan §10 the agent runs non-root, binds
// loopback by default, and exposes the serve port.
func NewImageConfig(agent effective.Agent, platform *specs.Platform) *specs.Image {
	img := &specs.Image{
		Platform: *platform,
	}
	img.RootFS.Type = "layers"

	img.Config.User = "1000:1000" // NON-ROOT (plan §10)
	img.Config.WorkingDir = "/"
	img.Config.Entrypoint = []string{utils.ServeBinary}
	img.Config.Cmd = []string{"--config", abi.Path}

	img.Config.Env = []string{
		"PATH=" + utils.AgentKitRoot + "/bin:" + system.DefaultPathEnv(utils.PlatformLinux),
		"AGENTKIT_BIND=127.0.0.1", // loopback default; 0.0.0.0 requires AGENTKIT_AUTH_TOKEN
		"PYTHONUNBUFFERED=1",
	}

	img.Config.ExposedPorts = map[string]struct{}{
		fmt.Sprintf("%d/tcp", agent.Expose.Port): {},
	}
	if agent.Expose.Port == utils.DefaultPort {
		// Foundry protocol mode defaults to 8088 for Hosted Agent compatibility.
		// Expose it in the image metadata too so docker -P and platforms that
		// inspect OCI ExposedPorts can discover the port the process uses.
		img.Config.ExposedPorts[fmt.Sprintf("%d/tcp", utils.DefaultFoundryPort)] = struct{}{}
	}

	capabilities := ""
	if runtimeSpec, ok := runtimes.RuntimeByName(agent.Runtime); ok {
		capabilities = strings.Join(runtimeSpec.Capabilities, ",")
	}

	generatedLabels := map[string]string{
		// Current AgentKit label namespace.
		config.ImageLabelNativeRuntime: agent.Runtime,
		config.ImageLabelNativeName:    agent.Metadata.Name,
		config.ImageLabelNativeABI:     abi.Version,
		// Cross-orchestrator metadata consumed by Orka and other registries.
		config.ImageLabelPortableABI:          abi.Version,
		config.ImageLabelPortableRuntime:      agent.Runtime,
		config.ImageLabelPortableProtocols:    imageProtocols,
		config.ImageLabelPortableCapabilities: capabilities,
		config.ImageLabelOrkaHarnessVersion:   orkaHarnessVersion,
		config.ImageLabelOCITitle:             agent.Metadata.Name,
	}

	img.Config.Labels = make(map[string]string, len(agent.Metadata.Labels)+len(generatedLabels))
	for k, v := range agent.Metadata.Labels {
		img.Config.Labels[k] = v
	}
	// Generated identity and capability labels are applied last so they remain
	// authoritative even if an invalid effective.Agent bypasses config validation.
	for k, v := range generatedLabels {
		img.Config.Labels[k] = v
	}

	return img
}

// NewComposedImageConfig preserves a narrow, safe subset of the selected base
// image metadata while keeping AgentKit authoritative over process execution.
func NewComposedImageConfig(agent effective.Agent, platform *specs.Platform, composition Composition) *specs.Image {
	img := NewImageConfig(agent, platform)
	img.Config.User = composition.User

	baseEnv := allowedBaseEnv(composition.BaseImage.Config.Env)
	basePath := system.DefaultPathEnv(utils.PlatformLinux)
	for _, entry := range baseEnv {
		if key, value, ok := strings.Cut(entry, "="); ok && strings.EqualFold(key, "PATH") {
			basePath = value
			break
		}
	}
	img.Config.Env = mergeEnv(baseEnv, []string{
		"PATH=" + utils.AgentKitRoot + "/bin:" + basePath,
		"AGENTKIT_BIND=127.0.0.1",
		"HOME=/tmp",
		"PYTHONPATH=" + utils.AgentKitRoot + "/packages",
		"PYTHONUNBUFFERED=1",
		"SHELL=" + utils.AgentKitRoot + "/bin/sh",
		"npm_config_script_shell=" + utils.AgentKitRoot + "/bin/sh",
	})

	for key, value := range composition.BaseImage.Config.Labels {
		if _, authoritative := img.Config.Labels[key]; !authoritative {
			img.Config.Labels[key] = value
		}
	}
	img.Config.Labels["org.opencontainers.image.base.name"] = composition.BaseName
	img.Config.Labels["org.opencontainers.image.base.digest"] = composition.BaseDigest
	img.Config.Labels[utils.LabelPrefix+".harness.name"] = composition.HarnessName
	img.Config.Labels[utils.LabelPrefix+".harness.digest"] = composition.HarnessDigest

	return img
}

func allowedBaseEnv(env []string) []string {
	out := make([]string, 0, len(env))
	for _, entry := range env {
		key, _, ok := strings.Cut(entry, "=")
		if !ok {
			continue
		}
		upper := strings.ToUpper(key)
		if upper == "PATH" ||
			upper == "LANG" ||
			upper == "LANGUAGE" ||
			upper == "TZ" ||
			upper == "HTTP_PROXY" ||
			upper == "HTTPS_PROXY" ||
			upper == "NO_PROXY" ||
			upper == "SSL_CERT_FILE" ||
			upper == "SSL_CERT_DIR" ||
			strings.HasPrefix(upper, "LC_") {
			out = append(out, entry)
		}
	}
	return out
}

func mergeEnv(base, overrides []string) []string {
	values := make(map[string]string, len(base)+len(overrides))
	order := make([]string, 0, len(base)+len(overrides))
	add := func(entry string) {
		key, _, ok := strings.Cut(entry, "=")
		if !ok {
			return
		}
		if _, exists := values[key]; !exists {
			order = append(order, key)
		}
		values[key] = entry
	}
	for _, entry := range base {
		add(entry)
	}
	for _, entry := range overrides {
		add(entry)
	}
	out := make([]string, 0, len(order))
	for _, key := range order {
		out = append(out, values[key])
	}
	return out
}
