from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from model_assets import ModelAsset, ModelAssetError, ensure_model_assets


class FakeS3:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.requests = []

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        self.requests.append((Bucket, Key))
        Path(Filename).write_bytes(self.payload)


def test_missing_asset_downloads_and_is_verified(tmp_path):
    payload = b"known test artifact"
    asset = ModelAsset("nested/weights.pth", "weights.pth", hashlib.sha256(payload).hexdigest())
    s3 = FakeS3(payload)

    paths = ensure_model_assets(
        bucket="private-model-bucket",
        prefix="models/v1",
        s3_client=s3,
        base_dir=tmp_path,
        assets=(asset,),
    )

    assert paths == (tmp_path / asset.destination,)
    assert paths[0].read_bytes() == payload
    assert s3.requests == [("private-model-bucket", "models/v1/weights.pth")]


def test_checksum_mismatch_does_not_install_file(tmp_path):
    asset = ModelAsset("nested/weights.pth", "weights.pth", "0" * 64)
    s3 = FakeS3(b"wrong content")

    with pytest.raises(ModelAssetError, match="SHA-256 verification failed"):
        ensure_model_assets(
            bucket="private-model-bucket",
            s3_client=s3,
            base_dir=tmp_path,
            assets=(asset,),
        )

    assert not (tmp_path / asset.destination).exists()
    assert list((tmp_path / "nested").iterdir()) == []


def test_existing_bad_asset_is_not_overwritten(tmp_path):
    asset = ModelAsset("weights.pth", "weights.pth", hashlib.sha256(b"expected").hexdigest())
    destination = tmp_path / asset.destination
    destination.write_bytes(b"local existing data")
    s3 = FakeS3(b"expected")

    with pytest.raises(ModelAssetError, match="left unchanged"):
        ensure_model_assets(
            bucket="private-model-bucket",
            s3_client=s3,
            base_dir=tmp_path,
            assets=(asset,),
        )

    assert destination.read_bytes() == b"local existing data"
    assert s3.requests == []


def test_missing_asset_without_bucket_explains_required_configuration(tmp_path):
    asset = ModelAsset("weights.pth", "weights.pth", "0" * 64)
    with pytest.raises(ModelAssetError, match="MODEL_ASSET_BUCKET"):
        ensure_model_assets(
            bucket="",
            base_dir=tmp_path,
            assets=(asset,),
        )
