package agent

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/orka-agents/agentkit/pkg/agentkit/abi"
	"github.com/orka-agents/agentkit/pkg/agentkit/config"
	"github.com/orka-agents/agentkit/pkg/agentkit/effective"
	"github.com/orka-agents/agentkit/pkg/agentkit/runtimes"
	"github.com/orka-agents/agentkit/pkg/utils"
)

const (
	testAgentName = "acme-support"
	testTeamLabel = "com.example/team"
	testTeamValue = "agentkit"
)

func imageAgent(runtime string, port int) effective.Agent {
	cfg := &config.AgentConfig{
		Metadata: config.Metadata{
			Name:   testAgentName,
			Labels: map[string]string{testTeamLabel: testTeamValue},
		},
		Runtime: runtime,
		Model: config.Model{
			Provider: utils.ProviderOpenAICompatible,
			BaseURL:  "https://api.openai.com/v1",
			Name:     "gpt-4o-mini",
		},
		Instructions: config.Source{Inline: "ignored after resolution"},
		Expose:       config.Expose{OpenAI: true, Port: port},
	}
	return effective.FromConfig(cfg, "resolved prompt")
}

func TestNewImageConfigUsesEffectiveAgentContract(t *testing.T) {
	agent := imageAgent(runtimes.MAFAlias, 0)
	img := NewImageConfig(agent, &specs.Platform{Architecture: utils.PlatformAMD64})

	if img.Config.User != "1000:1000" {
		t.Fatalf("User = %q, want non-root 1000:1000", img.Config.User)
	}
	if got := fmt.Sprint(img.Config.Cmd); got != fmt.Sprint([]string{"--config", abi.Path}) {
		t.Fatalf("Cmd = %v, want --config %s", img.Config.Cmd, abi.Path)
	}
	if _, ok := img.Config.ExposedPorts[fmt.Sprintf("%d/tcp", utils.DefaultPort)]; !ok {
		t.Fatalf("ExposedPorts = %#v, want default port %d", img.Config.ExposedPorts, utils.DefaultPort)
	}
	if _, ok := img.Config.ExposedPorts[fmt.Sprintf("%d/tcp", utils.DefaultFoundryPort)]; !ok {
		t.Fatalf("ExposedPorts = %#v, want Foundry default port %d", img.Config.ExposedPorts, utils.DefaultFoundryPort)
	}
	if got := img.Config.Labels[config.ImageLabelNativeRuntime]; got != runtimes.MAF {
		t.Fatalf("runtime label = %q, want canonical %q", got, runtimes.MAF)
	}
	if got := img.Config.Labels[config.ImageLabelNativeABI]; got != abi.Version {
		t.Fatalf("abi label = %q, want %q", got, abi.Version)
	}
	if got := img.Config.Labels[config.ImageLabelPortableABI]; got != abi.Version {
		t.Fatalf("portable abi label = %q, want %q", got, abi.Version)
	}
	if got := img.Config.Labels[config.ImageLabelPortableRuntime]; got != runtimes.MAF {
		t.Fatalf("portable runtime label = %q, want %q", got, runtimes.MAF)
	}
	if got := img.Config.Labels[config.ImageLabelPortableProtocols]; got != imageProtocols {
		t.Fatalf("protocols label = %q", got)
	}
	if got := img.Config.Labels[config.ImageLabelOrkaHarnessVersion]; got != orkaHarnessVersion {
		t.Fatalf("orka harness label = %q", got)
	}
	if got := img.Config.Labels[config.ImageLabelPortableCapabilities]; !strings.Contains(got, runtimes.CapabilityOrkaHarnessV1) {
		t.Fatalf("capabilities label = %q, want %s", got, runtimes.CapabilityOrkaHarnessV1)
	}
	if got := img.Config.Labels[testTeamLabel]; got != testTeamValue {
		t.Fatalf("custom label = %q, want %s", got, testTeamValue)
	}
}

func TestNewImageConfigAdvertisesRuntimeSelectableProtocols(t *testing.T) {
	for _, runtime := range runtimes.Runtimes {
		t.Run(runtime.Name, func(t *testing.T) {
			agent := imageAgent(runtime.Name, 0)
			agent.Metadata.Labels[config.ImageLabelPortableProtocols] = "user-controlled"
			img := NewImageConfig(agent, &specs.Platform{Architecture: utils.PlatformAMD64})
			const want = "openai,foundry,orka,agentsessions"
			if got := img.Config.Labels[config.ImageLabelPortableProtocols]; got != want {
				t.Fatalf("discoverable protocols = %q, want %q", got, want)
			}
		})
	}
}

func TestNewImageConfigPreservesTargetPlatformIdentity(t *testing.T) {
	platform := &specs.Platform{
		Architecture: "arm",
		OS:           "windows",
		OSVersion:    "10.0.20348.2113",
		OSFeatures:   []string{"win32k"},
		Variant:      "v7",
	}

	img := NewImageConfig(imageAgent(runtimes.MAF, 0), platform)

	if !reflect.DeepEqual(img.Platform, *platform) {
		t.Fatalf("Platform = %#v, want full target platform %#v", img.Platform, *platform)
	}
}

func TestNewImageConfigGeneratedLabelsOverrideMetadataLabels(t *testing.T) {
	agent := imageAgent(runtimes.MAFAlias, 0)
	runtimeSpec, ok := runtimes.RuntimeByName(agent.Runtime)
	if !ok {
		t.Fatalf("runtime %q is not registered", agent.Runtime)
	}
	expected := map[string]string{
		config.ImageLabelNativeRuntime:        runtimes.MAF,
		config.ImageLabelNativeName:           testAgentName,
		config.ImageLabelNativeABI:            abi.Version,
		config.ImageLabelPortableABI:          abi.Version,
		config.ImageLabelPortableRuntime:      runtimes.MAF,
		config.ImageLabelPortableProtocols:    imageProtocols,
		config.ImageLabelPortableCapabilities: strings.Join(runtimeSpec.Capabilities, ","),
		config.ImageLabelOrkaHarnessVersion:   orkaHarnessVersion,
		config.ImageLabelOCITitle:             testAgentName,
	}
	agent.Metadata.Labels = map[string]string{testTeamLabel: testTeamValue}
	for key := range expected {
		agent.Metadata.Labels[key] = "user-controlled"
	}

	img := NewImageConfig(agent, &specs.Platform{Architecture: utils.PlatformAMD64})

	for key, want := range expected {
		if got := img.Config.Labels[key]; got != want {
			t.Errorf("generated label %q = %q, want %q", key, got, want)
		}
	}
	if got := img.Config.Labels[testTeamLabel]; got != testTeamValue {
		t.Fatalf("unrelated user label = %q, want %q", got, testTeamValue)
	}
}

func TestNewImageConfigPreservesEffectivePort(t *testing.T) {
	agent := imageAgent("", 9090)
	img := NewImageConfig(agent, &specs.Platform{Architecture: utils.PlatformAMD64})

	if _, ok := img.Config.ExposedPorts["9090/tcp"]; !ok {
		t.Fatalf("ExposedPorts = %#v, want explicit 9090/tcp", img.Config.ExposedPorts)
	}
	if _, ok := img.Config.ExposedPorts[fmt.Sprintf("%d/tcp", utils.DefaultFoundryPort)]; ok {
		t.Fatalf("ExposedPorts = %#v, did not want Foundry default when explicit port is set", img.Config.ExposedPorts)
	}
}
