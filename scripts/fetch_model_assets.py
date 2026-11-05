#!/usr/bin/env python3
"""Prepare the two checksum-pinned weights required by PRISMORA AI inference."""

from model_assets import ModelAssetError, ensure_model_assets


def main() -> int:
    try:
        assets = ensure_model_assets()
    except ModelAssetError as exc:
        print(f"Model asset setup failed: {exc}")
        return 1

    for asset in assets:
        print(f"Verified model asset: {asset.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
