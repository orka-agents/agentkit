package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/moby/buildkit/solver/pb"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/orka-agents/agentkit/pkg/agentkit/runtimes"
)

func TestComposeAgentkit2LLBUsesBaseAndHarnessSources(t *testing.T) {
	platform := &specs.Platform{OS: "linux", Architecture: "arm64"}
	composition := Composition{
		BaseRef:       "registry.example/base@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		HarnessRef:    "registry.example/harness@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		ArtifactPath:  "/agentkit-harness",
		User:          "65532:65532",
		UID:           65532,
		GID:           65532,
		BaseImage:     &specs.Image{},
		BaseName:      "registry.example/base",
		BaseDigest:    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		HarnessName:   "registry.example/harness",
		HarnessDigest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
	}

	state, image, err := ComposeAgentkit2LLB(imageAgent(runtimes.PydanticAI, 0), composition, platform)
	if err != nil {
		t.Fatalf("ComposeAgentkit2LLB() error = %v", err)
	}
	if image.Config.User != composition.User {
		t.Fatalf("image user = %q, want %q", image.Config.User, composition.User)
	}
	def, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}

	var (
		sources     []string
		harnessCopy bool
	)
	for _, raw := range def.ToPB().Def {
		var op pb.Op
		if err := op.Unmarshal(raw); err != nil {
			t.Fatalf("unmarshal op: %v", err)
		}
		if source := op.GetSource(); source != nil {
			sources = append(sources, source.Identifier)
		}
		if file := op.GetFile(); file != nil {
			for _, action := range file.Actions {
				copyAction := action.GetCopy()
				if copyAction == nil {
					continue
				}
				if copyAction.Src == "/agentkit-harness" && copyAction.Dest == "/opt/agentkit" {
					harnessCopy = true
					if !copyAction.DirCopyContents {
						t.Error("harness copy does not copy directory contents")
					}
					if copyAction.Owner == nil ||
						copyAction.Owner.User == nil ||
						copyAction.Owner.User.GetByID() != 65532 ||
						copyAction.Owner.Group == nil ||
						copyAction.Owner.Group.GetByID() != 65532 {
						t.Errorf("harness copy owner = %#v, want 65532:65532", copyAction.Owner)
					}
				}
			}
		}
	}
	for _, want := range []string{"registry.example/base@", "registry.example/harness@"} {
		found := false
		for _, source := range sources {
			if strings.Contains(source, want) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("sources = %v, want source containing %q", sources, want)
		}
	}
	if !harnessCopy {
		t.Fatal("definition has no harness artifact copy")
	}
}
