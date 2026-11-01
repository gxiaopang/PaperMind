"""Pull a public Docker Hub image via host proxy, verify digests, then docker load.

Useful when Docker Desktop cannot use the host's configured HTTPS proxy.
No daemon settings are modified. Only linux/amd64 is supported by this helper.
"""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image", help="Docker Hub repository:tag, e.g. paradedb/paradedb:0.26.3-pg16"
    )
    args = parser.parse_args()
    repo, tag = args.image.rsplit(":", 1)
    if "/" not in repo:
        repo = "library/" + repo
    if not all(c.isalnum() or c in "/-_." for c in repo + tag):
        parser.error("Invalid image reference")
    root = Path("/tmp/papermind-images") / repo.replace("/", "_") / tag
    root.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        response = client.get(
            "https://auth.docker.io/token",
            params={
                "service": "registry.docker.io",
                "scope": f"repository:{repo}:pull",
            },
        )
        response.raise_for_status()
        headers = {
            "Authorization": f"Bearer {response.json()['token']}",
            "Accept": ", ".join(
                [
                    "application/vnd.oci.image.index.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.v2+json",
                ]
            ),
        }
        base = f"https://registry-1.docker.io/v2/{repo}"

        def manifest(ref):
            result = client.get(f"{base}/manifests/{ref}", headers=headers)
            result.raise_for_status()
            return result.json()

        data = manifest(tag)
        if "manifests" in data:
            chosen = next(
                row
                for row in data["manifests"]
                if row.get("platform", {}).get("os") == "linux"
                and row["platform"]["architecture"] == "amd64"
            )
            data = manifest(chosen["digest"])

        def blob(descriptor):
            digest = descriptor["digest"]
            target = root / digest.split(":")[1]
            if (
                target.exists()
                and hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
                == digest.split(":")[1]
            ):
                return target
            partial = target.with_suffix(".part")
            with client.stream("GET", f"{base}/blobs/{digest}", headers=headers) as stream:
                stream.raise_for_status()
                hasher = hashlib.sha256()
                with partial.open("wb") as file:
                    for chunk in stream.iter_bytes(1024 * 1024):
                        file.write(chunk)
                        hasher.update(chunk)
                if hasher.hexdigest() != digest.split(":")[1]:
                    raise ValueError("Layer SHA256 mismatch")
            partial.replace(target)
            print(
                f"Verified {digest[:22]} ({descriptor.get('size', 0) // 1048576} MiB)",
                flush=True,
            )
            return target

        config = blob(data["config"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            layers = list(pool.map(blob, data["layers"]))
        names = []
        for layer, descriptor in zip(layers, data["layers"], strict=True):
            target = layer.with_suffix(".tar")
            if not target.exists():
                opener = gzip.open if "gzip" in descriptor["mediaType"] else open
                with opener(layer, "rb") as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest, 1024 * 1024)
            names.append(target.name)
        manifest_file = root / "manifest.json"
        manifest_file.write_text(
            json.dumps([{"Config": config.name, "RepoTags": [args.image], "Layers": names}])
        )
        archive = root / "image.tar"
        with tarfile.open(archive, "w") as tar:
            for file in [config, manifest_file, *(root / name for name in names)]:
                tar.add(file, arcname=file.name)
        subprocess.run(["docker", "load", "-i", str(archive)], check=True)


if __name__ == "__main__":
    main()
