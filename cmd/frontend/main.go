package main

import (
	"os"

	"github.com/moby/buildkit/frontend/gateway/grpcclient"
	"github.com/moby/buildkit/util/appcontext"
	"github.com/moby/buildkit/util/bklog"
	"github.com/orka-agents/agentkit/pkg/agentkit/render"
	"github.com/orka-agents/agentkit/pkg/agentkit/toolbox"
	"github.com/orka-agents/agentkit/pkg/build"
	"github.com/sirupsen/logrus"
	"google.golang.org/grpc/grpclog"
)

func main() {
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "render":
			os.Exit(render.RunCLI(os.Args[2:], os.Stdout, os.Stderr))
		// Toolbox builds run these inside the frontend image itself.
		case toolbox.InstallCommand:
			os.Exit(toolbox.RunInstallCLI(os.Args[2:], os.Stdout, os.Stderr))
		case toolbox.VerifyCommand:
			os.Exit(toolbox.RunVerifyCLI(os.Args[2:], os.Stdout, os.Stderr))
		}
	}

	bklog.L.Logger.SetOutput(os.Stderr)
	grpclog.SetLoggerV2(grpclog.NewLoggerV2WithVerbosity(bklog.L.WriterLevel(logrus.InfoLevel), bklog.L.WriterLevel(logrus.WarnLevel), bklog.L.WriterLevel(logrus.ErrorLevel), 1))

	ctx := appcontext.Context()

	// Always run through the primary build router.
	if err := grpcclient.RunFromEnvironment(ctx, build.Build); err != nil {
		bklog.L.WithError(err).Fatal("error running frontend")
		os.Exit(1)
	}
}
