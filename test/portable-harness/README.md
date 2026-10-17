# Portable harness spike

This spike tests whether one Pydantic AI harness artifact can be copied into
unmodified Ubuntu, Fedora, UBI, and shell-less distroless base images.

The artifact can also exercise AgentKit's experimental production frontend:

```sh
make build-agentkit TAG=spike
make build-portable-harness-spike TAG=spike PLATFORM=linux/arm64
make build-composed-test-agent-spike TAG=spike PLATFORM=linux/arm64
```

That path independently resolves and digest-pins the base and harness, validates
the harness labels, copies only `/agentkit-harness`, and lets AgentKit generate
the final configuration and OCI metadata.

The harness artifact contains:

- a relocatable uv-managed CPython installation;
- AgentKit common and Pydantic AI packages sourced from the digest-pinned
  published adapter, with the local runtime code overlaid;
- launcher configuration so Python TLS uses the base image's system trust;
- Node.js, npm, and npx; and
- uv and uvx; and
- a harness-local BusyBox shell for tool launchers on shell-less bases; and
- native shared libraries required by bundled Python extensions but absent from
  the shell-less base.

The default matrix uses local Node and Python package fixtures that complete an
MCP-style `initialize` exchange through `npx` and `uvx`. This exercises package
installation and executable launch without registry access.

The composition fixture also accepts `RUNTIME_UID` and `RUNTIME_GID`. The
Ubuntu, Fedora, and UBI checks use `1000:1000`; the distroless check preserves
its native non-root identity `65532:65532`.

`HARNESS_LIBRARY_PATH` is empty for full distribution bases so their compatible
system libraries win. The distroless case sets it to `/opt/agentkit/lib` because
that base lacks libraries required by bundled native extensions and Node.js.
This distinction is deliberate: globally preferring a library copied from a
newer builder can break an older base such as UBI 9 through its glibc version.

Run the native-platform smoke matrix:

```sh
make test-portable-harness-spike
```

Override the target platform when needed:

```sh
make test-portable-harness-spike PLATFORM=linux/amd64
```

The default smoke is registry-independent after image construction. To also
exercise cold npm and PyPI-distributed MCP commands, enable the optional network
checks:

```sh
PORTABLE_HARNESS_NETWORK_TOOLS=1 make test-portable-harness-spike
```

Those checks require the container environment to reach npm and PyPI.

The enterprise-CA fixture currently runs on Ubuntu, Fedora, and UBI because it
installs a generated CA into each base's native trust store. The distroless case
still validates startup and bundled tooling but does not mutate its trust store.

This is feasibility scaffolding. It does not add `baseImage` or `harnessImage`
to the production AgentKit frontend.
