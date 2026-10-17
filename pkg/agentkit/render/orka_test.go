package render

import (
	"bytes"
	"strings"
	"testing"
)

const (
	testRuntimeName          = "fibey"
	testTargetFlag           = "--target"
	testNameFlag             = "--name"
	testExternalEndpointFlag = "--external-endpoint"
	testExampleEndpoint      = "http://example.invalid"
)

func TestOrkaAgentRuntimeExternalEndpoint(t *testing.T) {
	got, err := OrkaAgentRuntime(OrkaAgentRuntimeOptions{
		Name:             "fibey-agentkit",
		ExternalEndpoint: "http://fibey-agentkit.default.svc.cluster.local:8080",
	})
	if err != nil {
		t.Fatalf("OrkaAgentRuntime() error = %v", err)
	}
	want := `apiVersion: core.orka.ai/v1alpha1
kind: AgentRuntime
metadata:
  name: fibey-agentkit
spec:
  contractVersion: orka.harness.v1
  deployment:
    mode: external-endpoint
    endpoint: http://fibey-agentkit.default.svc.cluster.local:8080
  clientAuth:
    bearerTokenSecretRef:
      name: fibey-agentkit-harness-token
      key: token
  capabilities:
    toolExecutionModes:
    - observed
    supportsCancel: true
    supportsRuntimeSessions: true
`
	if string(got) != want {
		t.Fatalf("manifest mismatch\n got:\n%s\nwant:\n%s", got, want)
	}
}

func TestRunCLIRendersOrkaAgentRuntime(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunCLI([]string{
		testTargetFlag, TargetOrkaAgentRuntime,
		testExternalEndpointFlag, "http://fibey-agentkit.default.svc.cluster.local:8080",
		testNameFlag, "fibey-agentkit",
		"--auth-secret-name", "fibey-auth",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("RunCLI() code = %d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "kind: AgentRuntime") || !strings.Contains(out, "name: fibey-auth") {
		t.Fatalf("stdout = %s", out)
	}
}

func TestOrkaAgentRuntimeValidation(t *testing.T) {
	if _, err := OrkaAgentRuntime(OrkaAgentRuntimeOptions{Name: testRuntimeName}); err == nil {
		t.Fatal("expected missing external endpoint error")
	}
	if _, err := OrkaAgentRuntime(OrkaAgentRuntimeOptions{Name: testRuntimeName, Image: "ghcr.io/acme/fibey:latest"}); err == nil || !strings.Contains(err.Error(), "external endpoints only") {
		t.Fatalf("expected image unsupported error, got %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := RunCLI([]string{testTargetFlag, "other", testNameFlag, testRuntimeName, testExternalEndpointFlag, testExampleEndpoint}, &stdout, &stderr)
	if code == 0 || !strings.Contains(stderr.String(), "unsupported --target") {
		t.Fatalf("RunCLI() code=%d stderr=%q", code, stderr.String())
	}
}

func TestOrkaAgentRuntimeRejectsExternalEndpointsOutsideCRDPattern(t *testing.T) {
	const redactionMarker = "redaction-marker"
	invalid := []string{
		"example.com:8080",
		"ftp://example.com",
		"HTTP://example.com",
		"http:///missing-host",
		"http://:8080",
		"http://" + "user:" + redactionMarker + "@example.com",
		"http://example.com/path?q=" + redactionMarker,
		"http://example.com/path?",
		"http://example.com/path#" + redactionMarker,
		"http://example.com/path#",
		"http://example.com/path@segment",
		" http://example.com",
		"http://example.com ",
		"http://exa mple.com",
		"http://example.com\nnext",
	}
	for _, endpoint := range invalid {
		t.Run(endpoint, func(t *testing.T) {
			_, err := OrkaAgentRuntime(OrkaAgentRuntimeOptions{
				Name:             testRuntimeName,
				ExternalEndpoint: endpoint,
			})
			if err == nil {
				t.Fatalf("OrkaAgentRuntime() accepted invalid endpoint %q", endpoint)
			}
			if strings.Contains(err.Error(), redactionMarker) {
				t.Fatalf("validation error leaked endpoint material %q: %v", redactionMarker, err)
			}
		})
	}
}

func TestRunCLIExternalEndpointValidationRedactsCredentialBearingInput(t *testing.T) {
	markers := []string{"user-marker", "userinfo-marker", "query-marker", "anchor-marker"}
	endpoint := "https://" + markers[0] + ":" + markers[1] + "@example.com/path?q=" + markers[2] + "#" + markers[3]
	var stdout, stderr bytes.Buffer
	code := RunCLI([]string{
		testTargetFlag, TargetOrkaAgentRuntime,
		testNameFlag, testRuntimeName,
		testExternalEndpointFlag, endpoint,
	}, &stdout, &stderr)
	if code == 0 {
		t.Fatalf("RunCLI() accepted credential-bearing endpoint; stdout=%q", stdout.String())
	}
	if stdout.Len() != 0 {
		t.Fatalf("RunCLI() wrote stdout for invalid endpoint: %q", stdout.String())
	}
	for _, marker := range append([]string{endpoint}, markers...) {
		if strings.Contains(stderr.String(), marker) {
			t.Fatalf("RunCLI() leaked endpoint material %q in stderr: %q", marker, stderr.String())
		}
	}
	if !strings.Contains(stderr.String(), testExternalEndpointFlag) {
		t.Fatalf("RunCLI() stderr lacks generic endpoint guidance: %q", stderr.String())
	}
}
