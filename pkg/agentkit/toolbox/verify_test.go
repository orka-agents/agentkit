package toolbox

import (
	"bytes"
	"context"
	"strings"
	"testing"
	"time"
)

func TestVerifyRunsEveryCommand(t *testing.T) {
	var out bytes.Buffer
	if err := Verify(context.Background(), []string{"echo one", "echo two >&2"}, time.Minute, &out); err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if got, want := out.String(), "$ echo one\none\n$ echo two >&2\ntwo\n"; got != want {
		t.Fatalf("output = %q, want %q", got, want)
	}
}

func TestVerifyReportsFailuresWithOutput(t *testing.T) {
	var out bytes.Buffer
	glibc := "echo \"helm: /lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC_2.38' not found (required by helm)\" >&2; exit 1"
	err := Verify(context.Background(), []string{glibc, "echo fine", "exit 3"}, time.Minute, &out)
	if err == nil {
		t.Fatal("Verify succeeded, want failures")
	}
	for _, want := range []string{
		"2 of 3 verify commands failed",
		`"exit 3" failed with exit status 3`,
		`failed with exit status 1`,
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error = %v, want it to contain %q", err, want)
		}
	}
	for _, want := range []string{
		"GLIBC_2.38' not found",
		"hint: the tool needs a newer glibc than the runtime image has",
		"$ echo fine\nfine\n",
		`toolbox verify: "exit 3" failed with exit status 3`,
	} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output = %q, want it to contain %q", out.String(), want)
		}
	}
}

func TestVerifyTimesOut(t *testing.T) {
	var out bytes.Buffer
	start := time.Now()
	err := Verify(context.Background(), []string{"sleep 30"}, 200*time.Millisecond, &out)
	if err == nil || !strings.Contains(err.Error(), `"sleep 30" timed out after 200ms`) {
		t.Fatalf("Verify error = %v", err)
	}
	if elapsed := time.Since(start); elapsed > 10*time.Second {
		t.Fatalf("Verify took %s after the timeout", elapsed)
	}
}

func TestVerifyCLIRequiresCommands(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := RunVerifyCLI([]string{"--timeout", "1s"}, &stdout, &stderr); code != 2 {
		t.Fatalf("exit code = %d, stderr = %q", code, stderr.String())
	}
	if code := RunVerifyCLI([]string{"--", "true"}, &stdout, &stderr); code != 0 {
		t.Fatalf("exit code = %d, stderr = %q", code, stderr.String())
	}
	if code := RunVerifyCLI([]string{"--", "false"}, &stdout, &stderr); code != 1 {
		t.Fatalf("exit code = %d, stderr = %q", code, stderr.String())
	}
}
