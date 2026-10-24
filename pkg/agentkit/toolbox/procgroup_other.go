//go:build !unix

package toolbox

import (
	"os"
	"os/exec"
)

func startProcessGroup(*exec.Cmd) {}

func killProcessGroup(p *os.Process) error {
	if p == nil {
		return nil
	}
	return p.Kill()
}
