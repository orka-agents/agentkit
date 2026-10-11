# Contributing to AgentKit

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (for example, status check or comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repos using our CLA.

## Code of Conduct

Help us keep this project open and inclusive. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Security issues

Please do not report security vulnerabilities through public GitHub issues. See [SECURITY.md](SECURITY.md)
for security reporting guidance.

## Development and pull requests

See [README.md](README.md) for project setup and usage and the [development guide](docs/development.md)
for prerequisites, local Docker builds, and runtime-specific checks. Keep changes focused, add or
update tests and documentation where relevant, and describe the verification performed in your
pull request. Sign commits with a `Signed-off-by` line using `git commit -s`.

Run the checks relevant to your change from the repository root (Python checks use `uv`;
Go linting requires `golangci-lint`):

```sh
uv run --directory runtimes/common --extra dev pytest -q
make test  # Go tests with the race detector
make lint
```
