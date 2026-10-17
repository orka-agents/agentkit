"""Check a kind: Toolbox image exported as an OCI layout tarball.

Usage: toolbox-image-smoke.py <oci.tar> <metadata.name> <platform>...

Checks, for every platform: one layer holding only /opt/<name>, root-owned
0555 tools whose ELF headers match the platform, the generated ai.orka.toolbox
label, no runtime config, and an inventory whose digests match the files.
Prints one line per platform: <platform> <layer diff_ids> for comparisons.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
import tarfile

ELF_MACHINES = {"amd64": 62, "arm64": 183}
SOURCE_KEYS = {"download", "image", "path"}


def read_blob(oci: tarfile.TarFile, digest: str) -> bytes:
    algorithm, encoded = digest.split(":", 1)
    member = oci.extractfile(f"blobs/{algorithm}/{encoded}")
    assert member is not None, f"missing blob {digest}"
    data = member.read()
    assert hashlib.sha256(data).hexdigest() == encoded, f"blob {digest} is corrupt"
    return data


def platform_manifests(oci: tarfile.TarFile) -> dict[str, dict]:
    index = json.loads(oci.extractfile("index.json").read())
    manifests = {}
    pending = list(index["manifests"])
    while pending:
        descriptor = pending.pop()
        blob = json.loads(read_blob(oci, descriptor["digest"]))
        if "manifests" in blob:
            pending.extend(blob["manifests"])
            continue
        config = json.loads(read_blob(oci, blob["config"]["digest"]))
        platform = f'{config["os"]}/{config["architecture"]}'
        assert platform not in manifests, f"duplicate manifest for {platform}"
        manifests[platform] = {"manifest": blob, "config": config}
    return manifests


def check_platform(oci: tarfile.TarFile, name: str, platform: str, image: dict) -> list[str]:
    arch = platform.split("/")[1]
    mount = f"opt/{name}"
    config = image["config"]

    runtime = config.get("config", {})
    for key in ("Entrypoint", "Cmd", "User", "Env", "WorkingDir", "ExposedPorts"):
        assert not runtime.get(key), f"{platform}: toolbox image sets {key}: {runtime[key]!r}"
    labels = runtime.get("Labels", {})
    want_label = json.dumps(
        {"version": 1, "mountPath": f"/{mount}", "pathEntries": ["bin"]},
        separators=(",", ":"),
    )
    assert labels.get("ai.orka.toolbox") == want_label, f"{platform}: label {labels!r}"
    assert labels.get("org.opencontainers.image.title") == name, labels
    assert labels.get("com.example.purpose") == "e2e", labels

    layers = image["manifest"]["layers"]
    assert len(layers) == 1, f"{platform}: want one layer, got {len(layers)}"
    layer = read_blob(oci, layers[0]["digest"])
    if layers[0]["mediaType"].endswith("gzip"):
        layer = gzip.decompress(layer)

    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(layer)) as tar:
        entries = {m.name.rstrip("/"): m for m in tar.getmembers()}
        for path, member in entries.items():
            assert (member.uid, member.gid) == (0, 0), f"{path} is owned by {member.uid}:{member.gid}"
            if member.isdir():
                assert member.mode == 0o755, f"{path} mode {oct(member.mode)}"
            elif member.isfile():
                files[path] = tar.extractfile(member).read()
            else:
                raise AssertionError(f"{path} is neither a folder nor a regular file")
        assert set(entries) - set(files) == {"opt", mount, f"{mount}/bin"}, sorted(entries)

    inventory_path = f"{mount}/.toolbox.json"
    assert entries[inventory_path].mode == 0o444, oct(entries[inventory_path].mode)
    inventory = json.loads(files.pop(inventory_path))
    assert inventory["version"] == 1 and inventory["name"] == name, inventory
    assert inventory["platform"] == platform, inventory

    tools = [tool["name"] for tool in inventory["tools"]]
    assert sorted(files) == sorted(f"{mount}/bin/{tool}" for tool in tools), (sorted(files), tools)
    for tool in inventory["tools"]:
        path = f"{mount}/{tool['path']}"
        data = files[path]
        assert tool["path"] == f"bin/{tool['name']}", tool
        assert entries[path].mode == 0o555, f"{path} mode {oct(entries[path].mode)}"
        assert hashlib.sha256(data).hexdigest() == tool["sha256"], f"{path} digest"
        assert data[:4] == b"\x7fELF", f"{path} is not an ELF binary"
        machine = int.from_bytes(data[18:20], "little")
        assert machine == ELF_MACHINES[arch], f"{path} is ELF machine {machine}, not {arch}"
        source = tool["source"]
        assert set(source) <= SOURCE_KEYS and source, source
        if "download" in source:
            download = source["download"]
            assert download["url"].startswith("https://"), download
            if "extract" not in download:
                assert download["sha256"] == tool["sha256"], "plain download digest"
        else:
            assert "@sha256:" in source["image"] and source["path"].startswith("/"), source
    return config["rootfs"]["diff_ids"]


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    path, name, platforms = sys.argv[1], sys.argv[2], sys.argv[3:]
    with tarfile.open(path) as oci:
        manifests = platform_manifests(oci)
        assert sorted(manifests) == sorted(platforms), f"platforms {sorted(manifests)}"
        for platform in platforms:
            diff_ids = check_platform(oci, name, platform, manifests[platform])
            print(platform, ",".join(diff_ids))


if __name__ == "__main__":
    main()
