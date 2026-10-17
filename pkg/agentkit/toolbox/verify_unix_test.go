//go:build unix

package toolbox

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestVerifyStopsChildProcesses(t *testing.T) {
	dir := t.TempDir()
	timedOut := filepath.Join(dir, "timed-out")
	background := filepath.Join(dir, "background")
	start := time.Now()
	err := Verify(context.Background(), []string{
		// The shell waits on its child, so only the timeout ends it.
		fmt.Sprintf("sleep 30 & echo $! > %s; wait", timedOut),
		// The shell exits at once and leaves its child behind, still holding
		// the output pipe. That must neither fail nor delay verification.
		fmt.Sprintf("sleep 30 & echo $! > %s", background),
	}, time.Second, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "1 of 2 verify commands failed") || !strings.Contains(err.Error(), "timed out after 1s") {
		t.Fatalf("Verify error = %v", err)
	}
	if elapsed := time.Since(start); elapsed >= verifyDrainDelay {
		t.Fatalf("Verify took %s, want the background child killed when its shell exits", elapsed)
	}
	for _, pidFile := range []string{timedOut, background} {
		dt, err := os.ReadFile(pidFile)
		if err != nil {
			t.Fatal(err)
		}
		pid, err := strconv.Atoi(strings.TrimSpace(string(dt)))
		if err != nil {
			t.Fatal(err)
		}
		assertProcessGone(t, pid)
	}
}

// assertProcessGone waits for the killed process to be reaped by init.
func assertProcessGone(t *testing.T, pid int) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		err := syscall.Kill(pid, 0)
		if errors.Is(err, syscall.ESRCH) {
			return
		}
		if time.Now().After(deadline) {
			_ = syscall.Kill(pid, syscall.SIGKILL)
			t.Fatalf("child process %d is still running (signal 0: %v)", pid, err)
		}
		time.Sleep(50 * time.Millisecond)
	}
}
