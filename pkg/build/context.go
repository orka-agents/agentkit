package build

import (
	"archive/tar"
	"bytes"
	"context"
	"fmt"
	"net/url"
	"path"
	"strings"

	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/frontend/dockerfile/dfgitutil"
	"github.com/moby/buildkit/frontend/dockerui"
	"github.com/moby/buildkit/frontend/gateway/client"
	"github.com/orka-agents/agentkit/pkg/agentkit/config"
	"github.com/pkg/errors"
	"google.golang.org/grpc/status"
)

const httpContextProbeSize = 1024

type loadedAgentkitfile struct {
	config       *config.AgentConfig
	instructions contextFileReader
	// filename and source are the Agentkitfile as read, so validation
	// problems can be reported with their line.
	filename string
	source   []byte
}

type remoteContextIdentity struct {
	rawURL     string
	displayURL string
}

func newRemoteContextIdentity(rawURL string) remoteContextIdentity {
	return remoteContextIdentity{
		rawURL:     rawURL,
		displayURL: safeBuildContextDisplay(rawURL),
	}
}

func (i remoteContextIdentity) description(kind string) string {
	return kind + " " + i.displayURL
}

// loadAgentkitfile resolves the authored file and binds instruction reads to the
// same build context. Remote contexts retain one resolved reference for both the
// Agentkitfile and its relative files; local builds keep BuildKit's distinct
// dockerfile and context session inputs.
func loadAgentkitfile(ctx context.Context, c client.Client) (*loadedAgentkitfile, error) {
	opts := c.BuildOpts().Opts
	filename := opts[keyFilename]
	if filename == "" {
		filename = defaultAgentkitfileName
	}

	contextName := opts[localNameContext]
	if contextName == "" {
		return loadLocalAgentkitfile(ctx, c, opts, filename)
	}
	identity := newRemoteContextIdentity(contextName)

	keepGit := true
	gitState, isGit, gitErr := detectGitContext(identity, &keepGit)
	if isGit {
		if gitErr != nil {
			return nil, invalidGitContextError(identity, gitErr)
		}
		if gitState == nil {
			return nil, errors.Errorf("invalid Git build context %q: context detection returned no state", identity.displayURL)
		}
		return loadRemoteAgentkitfile(ctx, c, opts, filename, gitState, identity, "Git build context")
	}

	httpState, downloadedFilename, isHTTP := dockerui.DetectHTTPContext(identity.rawURL)
	if isHTTP {
		if err := validateHTTPContextURL(identity); err != nil {
			return nil, err
		}
		if httpState == nil || downloadedFilename == "" {
			return nil, errors.Errorf("invalid HTTP build context %q: context detection returned no source", identity.displayURL)
		}
		return loadHTTPAgentkitfile(ctx, c, opts, filename, identity, httpState, downloadedFilename)
	}

	return nil, errors.Errorf("unsupported build context %q: expected a Git or HTTP(S) context", identity.displayURL)
}

func loadLocalAgentkitfile(ctx context.Context, c client.Client, opts map[string]string, filename string) (*loadedAgentkitfile, error) {
	name := "load agentkitfile"
	if filename != defaultAgentkitfileName {
		name += " from " + filename
	}
	state := llb.Local(localNameDockerfile,
		llb.FollowPaths([]string{filename}),
		llb.SessionID(c.BuildOpts().SessionID),
		llb.SharedKeyHint(defaultAgentkitfileName),
		dockerui.WithInternalName(name),
	)
	ref, err := solveStateReference(ctx, c, &state, "local agentkitfile source")
	if err != nil {
		return nil, err
	}
	dt, err := readReferenceFile(ctx, ref, filename, "local agentkitfile source")
	if err != nil {
		return nil, errors.Wrap(err, "failed to read agentkitfile")
	}
	return parseLoadedAgentkitfile(filename, dt, opts, localContextReader{client: c})
}

func loadRemoteAgentkitfile(ctx context.Context, c client.Client, opts map[string]string, filename string, state *llb.State, identity remoteContextIdentity, kind string) (*loadedAgentkitfile, error) {
	description := identity.description(kind)
	ref, err := solveRemoteStateReference(ctx, c, state, description)
	if err != nil {
		return nil, err
	}
	reader := referenceContextReader{ref: ref, description: description}
	dt, err := reader.ReadFile(ctx, filename)
	if err != nil {
		return nil, errors.Wrap(err, "failed to read agentkitfile")
	}
	return parseLoadedAgentkitfile(filename, dt, opts, reader)
}

func loadHTTPAgentkitfile(ctx context.Context, c client.Client, opts map[string]string, filename string, identity remoteContextIdentity, state *llb.State, downloadedFilename string) (*loadedAgentkitfile, error) {
	description := identity.description("HTTP build context")
	rawRef, err := solveRemoteStateReference(ctx, c, state, description)
	if err != nil {
		return nil, err
	}

	header, err := readRemoteReferenceFile(ctx, rawRef, client.ReadRequest{
		Filename: downloadedFilename,
		Range:    &client.FileRange{Length: httpContextProbeSize},
	}, description)
	if err != nil {
		return nil, errors.Wrap(err, "failed to inspect HTTP build context")
	}

	if !isArchiveHeader(header) {
		dt, err := readRemoteReferenceFile(ctx, rawRef, client.ReadRequest{Filename: downloadedFilename}, description)
		if err != nil {
			return nil, errors.Wrap(err, "failed to read agentkitfile")
		}
		reader := unsupportedContextReader{err: errors.Errorf(
			"instructions.file is not supported for single-file HTTP context %q; use an HTTP archive, Git context, local context, or inline instructions",
			identity.displayURL,
		)}
		return parseLoadedAgentkitfile(downloadedFilename, dt, opts, reader)
	}

	unpacked := llb.Scratch().File(
		llb.Copy(*state, path.Join("/", downloadedFilename), "/", &llb.CopyInfo{AttemptUnpack: true}),
		dockerui.WithInternalName("unpack "+description),
	)
	return loadRemoteAgentkitfile(ctx, c, opts, filename, &unpacked, identity, "HTTP build context")
}

func parseLoadedAgentkitfile(filename string, dt []byte, opts map[string]string, reader contextFileReader) (*loadedAgentkitfile, error) {
	cfg, err := config.NewFromBytes(dt)
	if err != nil {
		return nil, errors.Wrap(err, "getting config")
	}
	if err := parseBuildArgs(opts, cfg); err != nil {
		return nil, errors.Wrap(err, "parsing build args")
	}
	return &loadedAgentkitfile{config: cfg, instructions: reader, filename: filename, source: dt}, nil
}

func solveStateReference(ctx context.Context, c client.Client, state *llb.State, description string) (client.Reference, error) {
	if state == nil {
		return nil, errors.Errorf("failed to resolve %s: context detection returned no state", description)
	}
	def, err := state.Marshal(ctx)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to marshal %s", description)
	}
	result, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
	if err != nil {
		return nil, errors.Wrapf(err, "failed to resolve %s", description)
	}
	if result == nil {
		return nil, errors.Errorf("failed to resolve %s: solve returned no result", description)
	}
	ref, err := result.SingleRef()
	if err != nil {
		return nil, errors.Wrapf(err, "failed to resolve %s reference", description)
	}
	if ref == nil {
		return nil, errors.Errorf("failed to resolve %s: solve returned no reference", description)
	}
	return ref, nil
}

// solveRemoteStateReference deliberately does not expose marshal/solve error
// causes. BuildKit or a transport may echo the raw source URL in those causes;
// callers instead receive the safe host/path display URL.
func solveRemoteStateReference(ctx context.Context, c client.Client, state *llb.State, description string) (client.Reference, error) {
	if state == nil {
		return nil, errors.Errorf("failed to resolve %s: context detection returned no state", description)
	}
	def, err := state.Marshal(ctx)
	if err != nil {
		return nil, errors.Errorf("failed to marshal %s", description)
	}

	result, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
	if err != nil {
		return nil, redactedRemoteError(err, "failed to resolve %s", description)
	}
	if result == nil {
		return nil, errors.Errorf("failed to resolve %s: solve returned no result", description)
	}
	ref, err := result.SingleRef()
	if err != nil {
		return nil, errors.Errorf("failed to resolve %s reference", description)
	}
	if ref == nil {
		return nil, errors.Errorf("failed to resolve %s: solve returned no reference", description)
	}
	return ref, nil
}

func readReferenceFile(ctx context.Context, ref client.Reference, filename, description string) ([]byte, error) {
	if ref == nil {
		return nil, errors.Errorf("failed to read %q from %s: context reference is nil", filename, description)
	}
	dt, err := ref.ReadFile(ctx, client.ReadRequest{Filename: filename})
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read %q from %s", filename, description)
	}
	return dt, nil
}

// readRemoteReferenceFile omits the underlying cause because gateway read
// errors may include the credential-bearing source URL.
func readRemoteReferenceFile(ctx context.Context, ref client.Reference, request client.ReadRequest, description string) ([]byte, error) {
	if ref == nil {
		return nil, errors.Errorf("failed to read %q from %s: context reference is nil", request.Filename, description)
	}
	dt, err := ref.ReadFile(ctx, request)
	if err != nil {
		return nil, redactedRemoteError(err, "failed to read %q from %s", request.Filename, description)
	}
	return dt, nil
}

func redactedRemoteError(err error, format string, args ...any) error {
	message := fmt.Sprintf(format, args...)
	switch {
	case errors.Is(err, context.Canceled):
		return errors.Wrap(context.Canceled, message)
	case errors.Is(err, context.DeadlineExceeded):
		return errors.Wrap(context.DeadlineExceeded, message)
	}
	if grpcStatus, ok := status.FromError(err); ok {
		return status.Error(grpcStatus.Code(), message)
	}
	return errors.New(message)
}

func validateHTTPContextURL(identity remoteContextIdentity) error {
	u, err := url.Parse(identity.rawURL)
	if err != nil {
		return errors.Errorf("invalid HTTP build context %q: malformed URL", identity.displayURL)
	}
	if u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
		return errors.Errorf("invalid HTTP build context %q: expected an absolute HTTP(S) URL", identity.displayURL)
	}
	return nil
}

func detectGitContext(identity remoteContextIdentity, keepGit *bool) (*llb.State, bool, error) {
	gitRef, isGit, err := dfgitutil.ParseGitRef(identity.rawURL)
	if err != nil {
		return nil, isGit, err
	}

	gitOpts := []llb.GitOption{
		llb.GitRef(gitRef.Ref),
		dockerui.WithInternalName("load git source " + identity.displayURL),
	}
	if gitRef.KeepGitDir != nil && *gitRef.KeepGitDir {
		gitOpts = append(gitOpts, llb.KeepGitDir())
	}
	if keepGit != nil && *keepGit {
		gitOpts = append(gitOpts, llb.KeepGitDir())
	}
	if gitRef.SubDir != "" {
		gitOpts = append(gitOpts, llb.GitSubDir(gitRef.SubDir))
	}
	if gitRef.Checksum != "" {
		gitOpts = append(gitOpts, llb.GitChecksum(gitRef.Checksum))
	}
	if gitRef.Submodules != nil && !*gitRef.Submodules {
		gitOpts = append(gitOpts, llb.GitSkipSubmodules())
	}
	if gitRef.MTime != "" {
		gitOpts = append(gitOpts, llb.GitMTime(gitRef.MTime))
	}
	if gitRef.FetchByCommit {
		gitOpts = append(gitOpts, llb.GitFetchByCommit())
	}

	state := llb.Git(gitRef.Remote, "", gitOpts...)
	return &state, true, nil
}

func invalidGitContextError(identity remoteContextIdentity, err error) error {
	return errors.Errorf("invalid Git build context %q: %s", identity.displayURL, safeGitContextErrorSummary(err))
}

func safeGitContextErrorSummary(err error) string {
	if err == nil {
		return "malformed Git context"
	}
	message := err.Error()
	switch {
	case strings.Contains(message, "ref conflicts"):
		return "ref conflicts"
	case strings.Contains(message, "subdir conflicts"):
		return "subdir conflicts"
	case strings.Contains(message, "branch conflicts with tag"):
		return "branch conflicts with tag"
	case strings.Contains(message, "multiple values"):
		return "invalid Git context query"
	case strings.Contains(message, "invalid keep-git-dir value"):
		return "invalid keep-git-dir value"
	case strings.Contains(message, "invalid submodules value"):
		return "invalid submodules value"
	case strings.Contains(message, "invalid fetch-by-commit value"):
		return "invalid fetch-by-commit value"
	case strings.Contains(message, "invalid mtime value"):
		return "invalid mtime value"
	default:
		return "malformed Git context options"
	}
}

func safeBuildContextDisplay(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		if strings.HasPrefix(strings.ToLower(raw), "http://") || strings.HasPrefix(strings.ToLower(raw), "https://") {
			return "HTTP(S) remote context"
		}
		return "remote context"
	}

	scheme := strings.ToLower(u.Scheme)
	if u.Host != "" {
		u.User = nil
		u.RawQuery = ""
		u.ForceQuery = false
		u.Fragment = ""
		u.RawFragment = ""
		return u.String()
	}
	if scheme == "http" || scheme == "https" {
		return "HTTP(S) remote context"
	}
	return "remote context"
}

// isArchiveHeader mirrors BuildKit's HTTP-context archive probe so compressed
// tarballs and plain tar streams are unpacked before the Agentkitfile is read.
func isArchiveHeader(header []byte) bool {
	for _, magic := range [][]byte{
		{0x42, 0x5A, 0x68},                   // bzip2
		{0x1F, 0x8B, 0x08},                   // gzip
		{0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00}, // xz
	} {
		if len(header) >= len(magic) && bytes.Equal(magic, header[:len(magic)]) {
			return true
		}
	}

	tr := tar.NewReader(bytes.NewReader(header))
	_, err := tr.Next()
	return err == nil
}
