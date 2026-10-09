// Command agentsessions is a wire-only Go/Python interop fixture, not a replay proof.
package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"time"

	v1 "github.com/aramase/agentsessions/api/genpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

func run() error {
	if len(os.Args) != 3 {
		return fmt.Errorf("usage: agentsessions ADDRESS DESCRIPTOR_ID")
	}
	conn, err := grpc.NewClient(os.Args[1], grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	ctx = metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer interop-local-token")
	client := v1.NewHarnessClient(conn)
	descriptor, err := client.Describe(ctx, &v1.DescribeRequest{})
	if err != nil {
		return err
	}
	if descriptor.GetId() != os.Args[2] || len(descriptor.GetModels()) != 1 || descriptor.GetModels()[0] != "host-model" || len(descriptor.GetTools()) != 0 {
		return fmt.Errorf("unexpected descriptor")
	}
	caps := descriptor.GetCapabilities()
	if caps.GetResumability() != v1.Resumability_RESUMABILITY_STATELESS_REPLAY || !caps.GetForkSafe() || caps.GetStreaming() || caps.GetRequiresGpu() || caps.GetReasoningReplay() {
		return fmt.Errorf("unexpected capabilities")
	}
	stream, err := client.Connect(ctx)
	if err != nil {
		return err
	}
	if err := stream.Send(&v1.ControllerFrame{
		ExecutionId: "go-execution",
		Frame: &v1.ControllerFrame_Start{Start: &v1.Start{
			Config: []byte("\x00go-config\xff"),
			Inputs: []*v1.Message{{Role: "user", Parts: []*v1.Part{{Part: &v1.Part_Text{Text: &v1.TextPart{Text: "from Go"}}}}}},
		}},
	}); err != nil {
		return err
	}
	// Keep the send side open, as harnesswire does: EOF is a disconnect.
	output, err := stream.Recv()
	if err != nil {
		return err
	}
	if output.GetExecutionId() != "go-execution" || output.GetSchemaVersion() != 1 || output.GetKind() != v1.EventKind_EVENT_OUTPUT || output.GetMessage().GetRole() != "assistant" || len(output.GetMessage().GetParts()) != 1 || output.GetMessage().GetParts()[0].GetText().GetText() != "from Python" {
		return fmt.Errorf("unexpected output frame")
	}
	end, err := stream.Recv()
	if err != nil {
		return err
	}
	if end.GetExecutionId() != "go-execution" || end.GetKind() != v1.EventKind_EVENT_END || end.GetEnd().GetState() != "COMPLETED" || end.GetEnd().GetError() != nil {
		return fmt.Errorf("unexpected terminal frame")
	}
	if _, err = stream.Recv(); err != io.EOF {
		return fmt.Errorf("expected EOF after exactly one END, got %v", err)
	}
	fmt.Println("Go/Python Harness wire interoperable")
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
