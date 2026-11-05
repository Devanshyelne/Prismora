"""Fetch and verify the trained Prismora inference weights from private S3 storage."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelAsset:
    destination: str
    object_key: str
    sha256: str


MODEL_ASSETS = (
    ModelAsset(
        destination="vgg_normalised.pth",
        object_key="vgg_normalised.pth",
        sha256="804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a",
    ),
    ModelAsset(
        destination="experiment/balanced_v2/decoder_final.pth",
        object_key="balanced_v2/decoder_final.pth",
        sha256="11a5f1811a503388f9ee246610e1b573089493dad7fa1efd4838d603c9150378",
    ),
)


class ModelAssetError(RuntimeError):
    """Raised when an expected trained model asset cannot be trusted or fetched."""


class S3Downloader(Protocol):
    def download_file(self, Bucket: str, Key: str, Filename: str) -> None: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, asset: ModelAsset) -> None:
    actual = sha256_file(path)
    if actual != asset.sha256:
        raise ModelAssetError(
            f"SHA-256 verification failed for model asset {asset.destination}. "
            "The file was left unchanged."
        )


def ensure_model_assets(
    *,
    bucket: str | None = None,
    prefix: str | None = None,
    s3_client: S3Downloader | None = None,
    base_dir: Path = BASE_DIR,
    assets: tuple[ModelAsset, ...] = MODEL_ASSETS,
) -> tuple[Path, ...]:
    """Verify local weights or download missing files from a configured S3 bucket.

    Downloads are written beside their destination, SHA-256 checked, then linked
    into place atomically. Existing files are never replaced.
    """
    bucket = bucket or os.environ.get("MODEL_ASSET_BUCKET", "").strip()
    prefix = (prefix if prefix is not None else os.environ.get("MODEL_ASSET_PREFIX", "models"))
    prefix = prefix.strip("/")
    verified: list[Path] = []

    for asset in assets:
        destination = base_dir / asset.destination
        if destination.is_file():
            _verify(destination, asset)
            verified.append(destination)
            continue

        if not bucket:
            raise ModelAssetError(
                "Model weights are missing. Configure MODEL_ASSET_BUCKET and the "
                "AWS S3 credentials/region, or place the verified model files locally."
            )

        if s3_client is None:
            region = os.environ.get("AWS_DEFAULT_REGION")
            if not region:
                raise ModelAssetError(
                    "AWS_DEFAULT_REGION is required to download model weights from S3."
                )
            try:
                import boto3

                s3_client = boto3.client("s3", region_name=region)
            except Exception as exc:
                raise ModelAssetError("Could not initialize the S3 model-artifact client.") from exc

        object_key = "/".join(part for part in (prefix, asset.object_key) if part)
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)

        try:
            s3_client.download_file(bucket, object_key, str(temporary))
            _verify(temporary, asset)
            try:
                # Same-directory hard-link creation is atomic and refuses to replace files.
                os.link(temporary, destination)
            except FileExistsError:
                _verify(destination, asset)
            verified.append(destination)
        except ModelAssetError:
            raise
        except Exception as exc:
            raise ModelAssetError(
                f"Could not download model artifact {object_key} from the configured bucket."
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)

    return tuple(verified)
