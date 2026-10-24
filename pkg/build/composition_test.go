package build

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/moby/buildkit/client/llb/sourceresolver"
	"github.com/moby/buildkit/frontend/gateway/client"
	"github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/orka-agents/agentkit/pkg/agentkit/runtimes"
)

func TestResolveImageInputs(t *testing.T) {
	rc := &RuntimeConfig{Name: runtimes.PydanticAI, defaultAdapterRef: "default-adapter"}
	tests := []struct {
		name    string
		opts    map[string]string
		want    imageInputs
		wantErr string
	}{
		{
			name: "default adapter",
			want: imageInputs{adapterRef: "default-adapter"},
		},
		{
			name: "adapter override",
			opts: map[string]string{"build-arg:adapter": "custom-adapter"},
			want: imageInputs{adapterRef: "custom-adapter"},
		},
		{
			name: "composition",
			opts: map[string]string{
				"build-arg:baseImage":    "ubuntu:24.04",
				"build-arg:harnessImage": "harness:test",
			},
			want: imageInputs{
				baseRef:     "ubuntu:24.04",
				harnessRef:  "harness:test",
				composition: true,
			},
		},
		{
			name:    "missing harness",
			opts:    map[string]string{"build-arg:baseImage": "ubuntu:24.04"},
			wantErr: "baseImage and harnessImage must be provided together",
		},
		{
			name:    "missing base",
			opts:    map[string]string{"build-arg:harnessImage": "harness:test"},
			wantErr: "baseImage and harnessImage must be provided together",
		},
		{
			name: "legacy override conflicts",
			opts: map[string]string{
				"build-arg:adapter":      "custom-adapter",
				"build-arg:baseImage":    "ubuntu:24.04",
				"build-arg:harnessImage": "harness:test",
			},
			wantErr: "mutually exclusive",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := resolveImageInputs(tt.opts, rc)
			if tt.wantErr != "" {
				if err == nil || !containsError(err, tt.wantErr) {
					t.Fatalf("resolveImageInputs() error = %v, want substring %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("resolveImageInputs() error = %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("resolveImageInputs() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestValidateHarnessLabels(t *testing.T) {
	valid := map[string]string{
		harnessLabelVersion:  "1",
		harnessLabelRuntime:  runtimes.PydanticAI,
		harnessLabelArtifact: "/agentkit-harness",
	}
	if got, err := validateHarnessLabels(valid, runtimes.PydanticAI); err != nil || got != "/agentkit-harness" {
		t.Fatalf("validateHarnessLabels() = %q, %v", got, err)
	}

	tests := []struct {
		name    string
		mutate  func(map[string]string)
		runtime string
		wantErr string
	}{
		{
			name: "unsupported version",
			mutate: func(labels map[string]string) {
				labels[harnessLabelVersion] = "2"
			},
			runtime: runtimes.PydanticAI,
			wantErr: harnessLabelVersion,
		},
		{
			name:    "runtime mismatch",
			mutate:  func(map[string]string) {},
			runtime: runtimes.LangGraph,
			wantErr: "does not match",
		},
		{
			name: "relative artifact",
			mutate: func(labels map[string]string) {
				labels[harnessLabelArtifact] = "agentkit-harness"
			},
			runtime: runtimes.PydanticAI,
			wantErr: "absolute path",
		},
		{
			name: "root artifact",
			mutate: func(labels map[string]string) {
				labels[harnessLabelArtifact] = "/"
			},
			runtime: runtimes.PydanticAI,
			wantErr: "must not be /",
		},
		{
			name: "unclean artifact",
			mutate: func(labels map[string]string) {
				labels[harnessLabelArtifact] = "/agentkit/../harness"
			},
			runtime: runtimes.PydanticAI,
			wantErr: "must be clean",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			labels := make(map[string]string, len(valid))
			for key, value := range valid {
				labels[key] = value
			}
			tt.mutate(labels)
			if _, err := validateHarnessLabels(labels, tt.runtime); err == nil || !containsError(err, tt.wantErr) {
				t.Fatalf("validateHarnessLabels() error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestCompositionUser(t *testing.T) {
	tests := []struct {
		input    string
		wantUser string
		wantUID  int
		wantGID  int
		wantErr  bool
	}{
		{input: "", wantUser: "1000:1000", wantUID: 1000, wantGID: 1000},
		{input: "root", wantUser: "1000:1000", wantUID: 1000, wantGID: 1000},
		{input: "65532", wantUser: "65532:65532", wantUID: 65532, wantGID: 65532},
		{input: "1001:2001", wantUser: "1001:2001", wantUID: 1001, wantGID: 2001},
		{input: "nonroot", wantErr: true},
		{input: "1000:group", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			user, uid, gid, err := compositionUser(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Fatal("compositionUser() error = nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("compositionUser() error = %v", err)
			}
			if user != tt.wantUser || uid != tt.wantUID || gid != tt.wantGID {
				t.Fatalf("compositionUser() = %q, %d, %d; want %q, %d, %d", user, uid, gid, tt.wantUser, tt.wantUID, tt.wantGID)
			}
		})
	}
}

type resolveImageClient struct {
	client.Client
	configs   map[string]specs.Image
	resolved  []string
	platforms []*specs.Platform
}

func (c *resolveImageClient) ResolveImageConfig(_ context.Context, ref string, opt sourceresolver.Opt) (string, digest.Digest, []byte, error) {
	image, ok := c.configs[ref]
	if !ok {
		return "", "", nil, errors.New("unexpected ref")
	}
	config, err := json.Marshal(image)
	if err != nil {
		return "", "", nil, err
	}
	c.resolved = append(c.resolved, ref)
	c.platforms = append(c.platforms, opt.ImageOpt.Platform)
	return ref, digest.FromString(ref), config, nil
}

func TestResolveCompositionPinsAndValidatesImages(t *testing.T) {
	platform := &specs.Platform{OS: "linux", Architecture: "arm64"}
	c := &resolveImageClient{configs: map[string]specs.Image{
		"docker.io/library/base:test": {
			Config: specs.ImageConfig{User: "65532"},
		},
		"docker.io/library/harness:test": {
			Config: specs.ImageConfig{Labels: map[string]string{
				harnessLabelVersion:  "1",
				harnessLabelRuntime:  runtimes.PydanticAI,
				harnessLabelArtifact: "/agentkit-harness",
			}},
		},
	}}

	composition, err := resolveComposition(context.Background(), c, runtimes.PydanticAI, imageInputs{
		baseRef:     "base:test",
		harnessRef:  "harness:test",
		composition: true,
	}, platform)
	if err != nil {
		t.Fatalf("resolveComposition() error = %v", err)
	}
	if composition.User != "65532:65532" || composition.UID != 65532 || composition.GID != 65532 {
		t.Fatalf("composition user = %q (%d:%d)", composition.User, composition.UID, composition.GID)
	}
	if composition.ArtifactPath != "/agentkit-harness" {
		t.Fatalf("artifact path = %q", composition.ArtifactPath)
	}
	if composition.BaseRef != pinImageRef("docker.io/library/base:test", digest.FromString("docker.io/library/base:test").String()) {
		t.Fatalf("base ref = %q", composition.BaseRef)
	}
	if composition.HarnessRef != pinImageRef("docker.io/library/harness:test", digest.FromString("docker.io/library/harness:test").String()) {
		t.Fatalf("harness ref = %q", composition.HarnessRef)
	}
	if !reflect.DeepEqual(c.resolved, []string{"docker.io/library/base:test", "docker.io/library/harness:test"}) {
		t.Fatalf("resolved refs = %v", c.resolved)
	}
	for _, got := range c.platforms {
		if got != platform {
			t.Fatalf("resolved platform = %p, want %p", got, platform)
		}
	}
}

func containsError(err error, substring string) bool {
	return err != nil && strings.Contains(err.Error(), substring)
}
