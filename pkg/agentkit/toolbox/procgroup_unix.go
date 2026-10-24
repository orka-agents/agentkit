//go:build unix

package toolbox

import (
	"os"
	"os/exec"
	"syscall"
)

func startProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// killProcessGroup kills every process in the group p leads, including
// children that outlived p.
func killProcessGroup(p *os.Process) error {
	if p == nil {
		return nil
	}
	return syscall.Kill(-p.Pid, syscall.SIGKILL)
}
