package render

import (
	"flag"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
)

// targetFlags lists the flags each render target accepts beyond --target and
// --output, so a flag meant for another target fails instead of being ignored.
var targetFlags = map[string][]string{
	TargetOrkaAgentRuntime:   {"image", "name", "external-endpoint", "auth-secret-name", "auth-secret-key"},
	TargetFoundryHostedAgent: {},
	TargetBrokeredTools:      {"no-digest", "bare"},
}

var targets = []string{TargetOrkaAgentRuntime, TargetFoundryHostedAgent, TargetBrokeredTools}

// RunCLI implements `agentkit render` for the frontend binary. The BuildKit
// frontend path still runs when no subcommand is provided.
//
//	agentkit render --target orka-agentruntime --name NAME --external-endpoint URL
//	agentkit render --target foundry-hosted-agent PROFILE
//	agentkit render --target brokered-tools [--no-digest] [--bare] TOOL_CRD_FILE...
func RunCLI(args []string, stdout io.Writer, stderr io.Writer) int {
	fs := flag.NewFlagSet("agentkit render", flag.ContinueOnError)
	fs.SetOutput(stderr)
	target := fs.String("target", "", "render target ("+strings.Join(targets, ", ")+")")
	var output string
	fs.StringVar(&output, "output", "", "write rendered YAML to this file instead of stdout")
	fs.StringVar(&output, "o", "", "shorthand for --output")
	image := fs.String("image", "", "orka-agentruntime: agent image reference; not supported by current Orka external-endpoint CRD")
	name := fs.String("name", "", "orka-agentruntime: AgentRuntime metadata.name")
	externalEndpoint := fs.String("external-endpoint", "", "orka-agentruntime: external Orka harness endpoint")
	authSecretName := fs.String("auth-secret-name", "", "orka-agentruntime: Secret name containing the Orka harness bearer token")
	authSecretKey := fs.String("auth-secret-key", DefaultAuthSecretKey, "orka-agentruntime: Secret key containing the Orka harness bearer token")
	noDigest := fs.Bool("no-digest", false, "brokered-tools: omit schemaDigest fields")
	bare := fs.Bool("bare", false, "brokered-tools: emit only the list, not a top-level brokeredTools key")
	inputs, err := parseInterleaved(fs, args)
	if err != nil {
		return 2
	}
	allowed, ok := targetFlags[*target]
	if !ok {
		fmt.Fprintf(stderr, "agentkit render: unsupported --target %q (expected one of %s)\n", *target, strings.Join(targets, ", "))
		return 2
	}
	if err := checkTargetFlags(fs, *target, allowed); err != nil {
		fmt.Fprintf(stderr, "agentkit render: %v\n", err)
		return 2
	}

	var out []byte
	switch *target {
	case TargetOrkaAgentRuntime:
		if len(inputs) > 0 {
			err = fmt.Errorf("--target %s takes no file arguments", *target)
			break
		}
		out, err = OrkaAgentRuntime(OrkaAgentRuntimeOptions{
			Name:             *name,
			Image:            *image,
			ExternalEndpoint: *externalEndpoint,
			AuthSecretName:   *authSecretName,
			AuthSecretKey:    *authSecretKey,
		})
	case TargetFoundryHostedAgent:
		if len(inputs) != 1 {
			err = fmt.Errorf("--target %s takes exactly one profile file", *target)
			break
		}
		out, err = FoundryHostedAgentFromFile(inputs[0])
	case TargetBrokeredTools:
		out, err = BrokeredToolsFromFiles(inputs, !*noDigest, *bare)
	}
	if err == nil && output != "" {
		err = os.WriteFile(output, out, 0o600)
	} else if err == nil {
		_, err = stdout.Write(out)
	}
	if err != nil {
		fmt.Fprintf(stderr, "agentkit render: %v\n", err)
		return 2
	}
	return 0
}

// parseInterleaved parses flags that may appear before, between, or after
// positional file arguments, and returns the positional arguments in order.
// Everything after a standalone `--` is positional, but `--` may still be the
// value of a flag that takes one.
func parseInterleaved(fs *flag.FlagSet, args []string) ([]string, error) {
	var flags, positional []string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--" {
			positional = append(positional, args[i+1:]...)
			break
		}
		if arg == "-" || !strings.HasPrefix(arg, "-") {
			positional = append(positional, arg)
			continue
		}
		flags = append(flags, arg)
		name := strings.TrimPrefix(strings.TrimPrefix(arg, "-"), "-")
		if f := fs.Lookup(name); f != nil && !isBoolFlag(f) && i+1 < len(args) {
			i++
			flags = append(flags, args[i])
		}
	}
	if err := fs.Parse(flags); err != nil {
		return nil, err
	}
	return positional, nil
}

func isBoolFlag(f *flag.Flag) bool {
	b, ok := f.Value.(interface{ IsBoolFlag() bool })
	return ok && b.IsBoolFlag()
}

func checkTargetFlags(fs *flag.FlagSet, target string, allowed []string) error {
	var unsupported []string
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "target", "output", "o":
			return
		}
		if !slices.Contains(allowed, f.Name) {
			unsupported = append(unsupported, "--"+f.Name)
		}
	})
	if len(unsupported) > 0 {
		return fmt.Errorf("--target %s does not accept %s", target, strings.Join(unsupported, ", "))
	}
	return nil
}
