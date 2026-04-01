#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


DEFAULT_MODELS = [
    "facebook/mask2former-swin-base-coco-panoptic",
    "facebook/mask2former-swin-base-ade-semantic"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-download Mask2Former models and store them in the local cache.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=[],
        help="Model id to cache (can be passed multiple times).",
    )
    parser.add_argument(
        "--include-ade",
        action="store_true",
        help="Also cache facebook/mask2former-swin-base-ade-semantic.",
    )
    return parser.parse_args()


def cache_model(model_id: str) -> bool:
    print(f"[cache] Downloading processor for {model_id} ...")
    AutoImageProcessor.from_pretrained(model_id)

    print(f"[cache] Downloading model weights for {model_id} ...")
    Mask2FormerForUniversalSegmentation.from_pretrained(model_id)

    print(f"[cache] Ready: {model_id}")
    return True


def main() -> int:
    args = parse_args()

    model_ids = list(DEFAULT_MODELS)
    model_ids.extend(args.models)
    if args.include_ade:
        model_ids.append("facebook/mask2former-swin-base-ade-semantic")

    # Keep order stable while removing duplicates.
    deduped = []
    seen = set()
    for model_id in model_ids:
        model_id = str(model_id).strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        deduped.append(model_id)

    if not deduped:
        print("No model ids provided.")
        return 1

    failures = 0
    for model_id in deduped:
        try:
            cache_model(model_id)
        except Exception as exc:
            failures += 1
            print(f"[cache] Failed: {model_id} -> {exc}")

    if failures:
        print(f"Finished with {failures} failure(s).")
        return 1

    print("All models cached successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
