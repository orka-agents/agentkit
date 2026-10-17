package toolbox

import (
	"encoding/json"
	"fmt"
	pathpkg "path"

	"github.com/containerd/platforms"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/orka-agents/agentkit/pkg/agentkit/config"
)

const (
	// InventoryFile lists the toolbox's tools, relative to its mount path.
	InventoryFile = ".toolbox.json"

	toolboxFormatVersion = 1
)

// Label is the value of the ai.orka.toolbox image label (version 1).
type Label struct {
	Version     int      `json:"version"`
	MountPath   string   `json:"mountPath"`
	PathEntries []string `json:"pathEntries"`
}

// NewImageConfig builds the config of a FROM scratch toolbox image. It has no
// entrypoint, user, environment, or agent labels: Orka never runs a toolbox,
// it only mounts its files.
func NewImageConfig(cfg *config.ToolboxConfig, platform specs.Platform) (*specs.Image, error) {
	label, err := json.Marshal(Label{
		Version:     toolboxFormatVersion,
		MountPath:   cfg.MountPath(),
		PathEntries: []string{config.ToolboxPathEntry},
	})
	if err != nil {
		return nil, err
	}

	img := &specs.Image{Platform: platform}
	img.RootFS.Type = "layers"
	img.Config.Labels = make(map[string]string, len(cfg.Metadata.Labels)+2)
	for k, v := range cfg.Metadata.Labels {
		img.Config.Labels[k] = v
	}
	// Generated labels are applied last so they stay authoritative.
	img.Config.Labels[config.ImageLabelOrkaToolbox] = string(label)
	img.Config.Labels[config.ImageLabelOCITitle] = cfg.Metadata.Name
	return img, nil
}

type inventory struct {
	Version  int             `json:"version"`
	Name     string          `json:"name"`
	Platform string          `json:"platform"`
	Tools    []inventoryTool `json:"tools"`
}

type inventoryTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Path        string          `json:"path"`
	SHA256      string          `json:"sha256"`
	Source      inventorySource `json:"source"`
}

type inventorySource struct {
	Download *inventoryDownload `json:"download,omitempty"`
	Image    string             `json:"image,omitempty"`
	Path     string             `json:"path,omitempty"`
}

type inventoryDownload struct {
	URL     string `json:"url"`
	SHA256  string `json:"sha256"`
	Extract string `json:"extract,omitempty"`
}

// Inventory renders InventoryFile for one platform: each tool's name,
// description, source, and the SHA-256 of the installed file.
func Inventory(cfg *config.ToolboxConfig, tools []Tool, platform specs.Platform) ([]byte, error) {
	inv := inventory{
		Version:  toolboxFormatVersion,
		Name:     cfg.Metadata.Name,
		Platform: platforms.Format(platform),
		Tools:    make([]inventoryTool, 0, len(tools)),
	}
	for _, tool := range tools {
		if tool.Result.SHA256 == "" {
			return nil, fmt.Errorf("tool %q has no installed SHA-256", tool.Name)
		}
		entry := inventoryTool{
			Name:        tool.Name,
			Description: tool.Description,
			Path:        pathpkg.Join(config.ToolboxPathEntry, tool.Name),
			SHA256:      tool.Result.SHA256,
		}
		if tool.Source.Image != "" {
			entry.Source = inventorySource{Image: tool.Source.Image, Path: tool.Source.Path}
		} else {
			download, ok := tool.Source.DownloadFor(platform)
			if !ok {
				return nil, fmt.Errorf("tool %q has no download for %s", tool.Name, inv.Platform)
			}
			entry.Source = inventorySource{Download: &inventoryDownload{
				URL:     download.URL,
				SHA256:  download.SHA256,
				Extract: download.Extract,
			}}
		}
		inv.Tools = append(inv.Tools, entry)
	}
	dt, err := json.MarshalIndent(inv, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(dt, '\n'), nil
}
