// Command agentkit-toolbox is the helper a kind: Toolbox build runs from the
// AgentKit frontend image: `install` takes one tool file and `verify` runs the
// verify stage. The image ships one static build per supported architecture.
package main

import (
	"fmt"
	"os"

	"github.com/orka-agents/agentkit/pkg/agentkit/toolbox"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: agentkit-toolbox %s|%s [flags]\n", toolbox.InstallCommand, toolbox.VerifyCommand)
		os.Exit(2)
	}
	switch os.Args[1] {
	case toolbox.InstallCommand:
		os.Exit(toolbox.RunInstallCLI(os.Args[2:], os.Stdout, os.Stderr))
	case toolbox.VerifyCommand:
		os.Exit(toolbox.RunVerifyCLI(os.Args[2:], os.Stdout, os.Stderr))
	default:
		fmt.Fprintf(os.Stderr, "agentkit-toolbox: unknown command %q\n", os.Args[1])
		os.Exit(2)
	}
}
