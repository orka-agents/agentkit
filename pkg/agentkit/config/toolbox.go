package config

import (
	"errors"
	"fmt"
	"net/url"
	pathpkg "path"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"github.com/containerd/platforms"
	"github.com/distribution/reference"
	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/orka-agents/agentkit/pkg/utils"
)

const (
	// ToolboxMountRoot is the parent folder of every toolbox mount path.
	ToolboxMountRoot = "/opt"
	// ToolboxPathEntry is the folder, relative to the mount path, holding tools.
	ToolboxPathEntry = "bin"

	toolboxNameMaxLength = 63
)

var (
	toolboxNamePattern = regexp.MustCompile(`^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`)
	sha256HexPattern   = regexp.MustCompile(`^[a-f0-9]{64}$`)

	// orkaReservedToolboxNames are /opt/<name> folders Orka's runtime images
	// already use. Orka rejects a toolbox mounted over one of them.
	orkaReservedToolboxNames = [...]string{
		"claude", "claude-agent-acp", "codex", "codex-acp", "copilot", "opencode", "ripgrep",
	}
)

// ToolboxConfig is the parsed agentkitfile for kind: Toolbox. A toolbox is a
// FROM scratch image whose tools live under /opt/<metadata.name>/bin so Orka
// can mount it read-only into the pods of its built-in coding agents.
type ToolboxConfig struct {
	APIVersion string      `yaml:"apiVersion"`
	Kind       string      `yaml:"kind"`
	Metadata   Metadata    `yaml:"metadata"`
	Spec       ToolboxSpec `yaml:"spec"`
}

// ToolboxSpec lists the tools and the optional verify stage.
type ToolboxSpec struct {
	Tools  []ToolboxTool  `yaml:"tools"`
	Verify *ToolboxVerify `yaml:"verify,omitempty"`
}

// ToolboxTool installs one file as /opt/<metadata.name>/bin/<name>.
type ToolboxTool struct {
	Name        string            `yaml:"name"`
	Description string            `yaml:"description,omitempty"`
	Source      ToolboxToolSource `yaml:"source"`
}

// ToolboxToolSource is a tagged union: exactly one of Download or Image is set.
type ToolboxToolSource struct {
	// Download maps a platform such as linux/arm64 to its checksum-pinned file.
	Download map[string]ToolboxDownload `yaml:"download,omitempty"`
	// Image is a digest-pinned image reference holding the tool at Path.
	Image string `yaml:"image,omitempty"`
	// Path is the absolute path of the tool inside Image.
	Path string `yaml:"path,omitempty"`
}

// ToolboxDownload is one platform's download. Extract names the single regular
// file to take from an archive; without it the download itself is the tool.
type ToolboxDownload struct {
	URL     string `yaml:"url"`
	SHA256  string `yaml:"sha256"`
	Extract string `yaml:"extract,omitempty"`
}

// ToolboxVerify runs smoke commands inside a runtime image with the toolbox
// mounted. It only runs for `--target verify` and never changes the image.
type ToolboxVerify struct {
	RuntimeImage string   `yaml:"runtimeImage"`
	Commands     []string `yaml:"commands"`
}

// MountPath is the folder the toolbox installs under and Orka mounts.
func (c *ToolboxConfig) MountPath() string {
	return pathpkg.Join(ToolboxMountRoot, c.Metadata.Name)
}

// Digest returns the download's checksum as an OCI digest.
func (d ToolboxDownload) Digest() digest.Digest {
	return digest.NewDigestFromEncoded(digest.SHA256, d.SHA256)
}

// DownloadFor returns the download declared for platform. Keys match after
// normalization, so linux/aarch64 and linux/arm64 name the same platform.
func (s ToolboxToolSource) DownloadFor(platform specs.Platform) (ToolboxDownload, bool) {
	want := platforms.Format(platforms.Normalize(platform))
	for key, download := range s.Download {
		if got, ok := normalizedPlatformKey(key); ok && got == want {
			return download, true
		}
	}
	return ToolboxDownload{}, false
}

// DownloadPlatforms returns the normalized platforms the source can download.
func (s ToolboxToolSource) DownloadPlatforms() []string {
	out := make([]string, 0, len(s.Download))
	for key := range s.Download {
		if normalized, ok := normalizedPlatformKey(key); ok {
			out = append(out, normalized)
		}
	}
	sort.Strings(out)
	return out
}

func normalizedPlatformKey(key string) (string, bool) {
	// A bare "linux" would silently default to the builder's architecture.
	if !strings.Contains(key, "/") {
		return "", false
	}
	p, err := platforms.Parse(key)
	if err != nil {
		return "", false
	}
	return platforms.Format(platforms.Normalize(p)), true
}

// Validate reports every problem with the toolbox at once.
func (c *ToolboxConfig) Validate() error {
	var errs []error
	add := func(format string, args ...any) {
		errs = append(errs, fmt.Errorf(format, args...))
	}

	if c.APIVersion == "" {
		add("apiVersion is not defined")
	} else if c.APIVersion != utils.APIv1alpha1 {
		add("apiVersion %q is not supported (expected %q)", c.APIVersion, utils.APIv1alpha1)
	}
	if c.Kind != utils.KindToolbox {
		add("kind %q is not supported (expected %q)", c.Kind, utils.KindToolbox)
	}

	switch name := c.Metadata.Name; {
	case name == "":
		add("metadata.name is required")
	case !isToolboxName(name):
		add("metadata.name %q must be 1-%d lowercase letters, digits, or '-', starting and ending with a letter or digit", name, toolboxNameMaxLength)
	case isOrkaReservedToolboxName(name):
		add("metadata.name %q is reserved: Orka's runtime images already use %s", name, c.MountPath())
	}
	validateMetadataLabels(add, c.Metadata.Labels)

	if len(c.Spec.Tools) == 0 {
		add("spec.tools must list at least one tool")
	}
	seen := make(map[string]int, len(c.Spec.Tools))
	for i, tool := range c.Spec.Tools {
		field := fmt.Sprintf("spec.tools[%d]", i)
		switch {
		case tool.Name == "":
			add("%s.name is required", field)
		case !isToolboxName(tool.Name):
			add("%s.name %q must be 1-%d lowercase letters, digits, or '-', starting and ending with a letter or digit", field, tool.Name, toolboxNameMaxLength)
		default:
			if first, dup := seen[tool.Name]; dup {
				add("%s.name %q duplicates spec.tools[%d].name", field, tool.Name, first)
			} else {
				seen[tool.Name] = i
			}
		}
		validateToolboxSource(add, field+".source", tool.Source)
	}

	if v := c.Spec.Verify; v != nil {
		if v.RuntimeImage == "" {
			add("spec.verify.runtimeImage is required")
		} else {
			validateDigestPinnedImage(add, "spec.verify.runtimeImage", v.RuntimeImage)
		}
		if len(v.Commands) == 0 {
			add("spec.verify.commands must list at least one command")
		}
		for i, command := range v.Commands {
			if strings.TrimSpace(command) == "" {
				add("spec.verify.commands[%d] must not be empty", i)
			} else if strings.ContainsRune(command, 0) {
				add("spec.verify.commands[%d] must not contain NUL", i)
			}
		}
	}

	return errors.Join(errs...)
}

func validateToolboxSource(add func(string, ...any), field string, src ToolboxToolSource) {
	hasDownload := len(src.Download) > 0
	hasImage := src.Image != ""
	switch {
	case hasDownload && hasImage:
		add("%s must set exactly one of download or image", field)
	case !hasDownload && !hasImage:
		add("%s must set download or image", field)
	}

	if hasImage {
		validateDigestPinnedImage(add, field+".image", src.Image)
		if src.Path == "" {
			add("%s.path is required with %s.image", field, field)
		} else if !isCleanAbsolutePath(src.Path) {
			add("%s.path %q must be a clean absolute file path", field, src.Path)
		}
	} else if src.Path != "" {
		add("%s.path is only valid with %s.image", field, field)
	}

	keys := make([]string, 0, len(src.Download))
	for key := range src.Download {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	normalized := make(map[string]string, len(keys))
	for _, key := range keys {
		download := src.Download[key]
		keyField := fmt.Sprintf("%s.download[%q]", field, key)

		if platform, ok := normalizedPlatformKey(key); !ok {
			add("%s: platform must be <os>/<arch>[/<variant>], for example linux/amd64", keyField)
		} else if !strings.HasPrefix(platform, utils.PlatformLinux+"/") {
			add("%s: platform must be a linux platform", keyField)
		} else if first, dup := normalized[platform]; dup {
			add("%s names the same platform as %q", keyField, first)
		} else {
			normalized[platform] = key
		}

		if download.URL == "" {
			add("%s.url is required", keyField)
		} else if u, err := url.Parse(download.URL); err != nil || u.Scheme != "https" || u.Host == "" {
			add("%s.url must be an absolute https URL", keyField)
		} else if u.User != nil {
			add("%s.url must not embed credentials", keyField)
		}
		if !sha256HexPattern.MatchString(download.SHA256) {
			add("%s.sha256 must be 64 lowercase hex characters", keyField)
		}
		if download.Extract != "" && !isCleanRelativePath(download.Extract) {
			add("%s.extract %q must be a clean relative file path inside the archive", keyField, download.Extract)
		}
	}
}

// validateDigestPinnedImage requires an image reference pinned by sha256
// digest, so the bytes a build uses can never change underneath it.
func validateDigestPinnedImage(add func(string, ...any), field, ref string) {
	named, err := reference.ParseNormalizedNamed(ref)
	if err != nil {
		add("%s %q is not a valid image reference", field, ref)
		return
	}
	canonical, ok := named.(reference.Canonical)
	if !ok || canonical.Digest().Algorithm() != digest.SHA256 {
		add("%s %q must be pinned by digest (<image>@sha256:<64 hex>), not a tag", field, ref)
	}
}

func isToolboxName(name string) bool {
	return len(name) <= toolboxNameMaxLength && toolboxNamePattern.MatchString(name)
}

func isOrkaReservedToolboxName(name string) bool {
	for _, reserved := range orkaReservedToolboxNames {
		if name == reserved {
			return true
		}
	}
	return false
}

func isCleanAbsolutePath(p string) bool {
	return strings.HasPrefix(p, "/") && p != "/" && pathpkg.Clean(p) == p && !hasControlCharacter(p)
}

func isCleanRelativePath(p string) bool {
	return p != "" && p != "." && !strings.HasPrefix(p, "/") && pathpkg.Clean(p) == p &&
		p != ".." && !strings.HasPrefix(p, "../") && !hasControlCharacter(p)
}

func hasControlCharacter(s string) bool {
	return strings.IndexFunc(s, unicode.IsControl) >= 0
}
