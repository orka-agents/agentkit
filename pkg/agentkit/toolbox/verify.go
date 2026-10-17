package toolbox

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"
)

const (
	// DefaultVerifyTimeout bounds each verify command.
	DefaultVerifyTimeout = time.Minute

	// verifyOutputTail is how much recent output Verify keeps to explain a failure.
	verifyOutputTail = 64 << 10
	// verifyWaitDelay bounds how long a timed-out command's children may hold
	// its output open.
	verifyWaitDelay = 5 * time.Second
)

// Verify runs each command with /bin/sh -c, streaming its output to out. It
// runs every command, then reports each one that failed or timed out.
func Verify(ctx context.Context, commands []string, timeout time.Duration, out io.Writer) error {
	var failed []string
	for _, command := range commands {
		fmt.Fprintf(out, "$ %s\n", command)
		tail := &tailBuffer{max: verifyOutputTail}
		err := runVerifyCommand(ctx, command, timeout, io.MultiWriter(out, tail))
		if err == nil {
			continue
		}
		failed = append(failed, fmt.Sprintf("%q %s", command, err))
		fmt.Fprintf(out, "toolbox verify: %q %s\n", command, err)
		if hint := verifyHint(tail.Bytes()); hint != "" {
			fmt.Fprintf(out, "hint: %s\n", hint)
		}
	}
	if len(failed) > 0 {
		return fmt.Errorf("%d of %d verify commands failed:\n  %s", len(failed), len(commands), strings.Join(failed, "\n  "))
	}
	return nil
}

func runVerifyCommand(ctx context.Context, command string, timeout time.Duration, out io.Writer) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "/bin/sh", "-c", command)
	cmd.Stdout = out
	cmd.Stderr = out
	cmd.WaitDelay = verifyWaitDelay
	// The command runs in its own process group so a timeout stops everything
	// it started, and nothing it left in the background outlives it.
	startProcessGroup(cmd)
	cmd.Cancel = func() error { return killProcessGroup(cmd.Process) }
	err := cmd.Run()
	_ = killProcessGroup(cmd.Process)
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return fmt.Errorf("timed out after %s", timeout)
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return fmt.Errorf("failed with exit status %d", exitErr.ExitCode())
	}
	if err != nil {
		return fmt.Errorf("could not run: %w", err)
	}
	return nil
}

// verifyHint names the usual cause of a tool that cannot start in the
// runtime image.
func verifyHint(output []byte) string {
	switch {
	case bytes.Contains(output, []byte("GLIBC_")) && bytes.Contains(output, []byte("not found")):
		return "the tool needs a newer glibc than the runtime image has; use a static build or one built for the runtime's glibc"
	case bytes.Contains(output, []byte("error while loading shared libraries")):
		return "the tool needs a shared library the runtime image does not have; use a static build"
	default:
		return ""
	}
}

// tailBuffer keeps the last max bytes written to it.
type tailBuffer struct {
	buf []byte
	max int
}

func (t *tailBuffer) Write(p []byte) (int, error) {
	t.buf = append(t.buf, p...)
	if over := len(t.buf) - t.max; over > 0 {
		t.buf = t.buf[over:]
	}
	return len(p), nil
}

func (t *tailBuffer) Bytes() []byte {
	return t.buf
}
