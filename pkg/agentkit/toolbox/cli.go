package toolbox

import (
	"context"
	"flag"
	"fmt"
	"io"
)

const (
	// InstallCommand and VerifyCommand are the agentkit-toolbox subcommands.
	InstallCommand = "install"
	VerifyCommand  = "verify"
)

// RunInstallCLI implements `agentkit-toolbox install`.
func RunInstallCLI(args []string, stdout io.Writer, stderr io.Writer) int {
	fs := flag.NewFlagSet("agentkit-toolbox "+InstallCommand, flag.ContinueOnError)
	fs.SetOutput(stderr)
	var req Request
	fs.StringVar(&req.Source, "src", "", "downloaded file, or file copied from an image")
	fs.StringVar(&req.Member, "member", "", "archive member to extract; empty installs --src itself")
	fs.StringVar(&req.Arch, "arch", "", "GOARCH the tool must be built for")
	fs.StringVar(&req.Dest, "dest", "", "directory receiving "+ToolFile+" and "+ResultFile)
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if req.Source == "" || req.Arch == "" || req.Dest == "" || fs.NArg() != 0 {
		fmt.Fprintf(stderr, "agentkit-toolbox %s: --src, --arch, and --dest are required\n", InstallCommand)
		return 2
	}
	result, err := Install(req, DefaultLimits)
	if err != nil {
		fmt.Fprintf(stderr, "agentkit-toolbox %s: %v\n", InstallCommand, err)
		return 1
	}
	fmt.Fprintf(stdout, "installed tool sha256:%s\n", result.SHA256)
	return 0
}

// RunVerifyCLI implements `agentkit-toolbox verify -- <command>...`.
func RunVerifyCLI(args []string, stdout io.Writer, stderr io.Writer) int {
	fs := flag.NewFlagSet("agentkit-toolbox "+VerifyCommand, flag.ContinueOnError)
	fs.SetOutput(stderr)
	timeout := fs.Duration("timeout", DefaultVerifyTimeout, "timeout for each command")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() == 0 || *timeout <= 0 {
		fmt.Fprintf(stderr, "agentkit-toolbox %s: a positive --timeout and at least one command are required\n", VerifyCommand)
		return 2
	}
	if err := Verify(context.Background(), fs.Args(), *timeout, stdout); err != nil {
		fmt.Fprintf(stderr, "agentkit-toolbox %s: %v\n", VerifyCommand, err)
		return 1
	}
	return 0
}
