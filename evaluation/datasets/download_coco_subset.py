"""Download a small subset of MS-COCO val2017 (images + captions) for the
cross-modal retrieval benchmark.

Dataset choice & provenance (documented per project requirement to justify the
autonomously-selected benchmark dataset):
  - MS-COCO (https://cocodataset.org) is a standard, widely-used benchmark for
    text<->image retrieval. Captions are licensed CC BY 4.0; images are sourced
    from Flickr under a mix of Creative Commons licenses (documented per-image at
    https://cocodataset.org/#termsofuse). This is the customary licensing basis
    under which COCO is used for non-commercial research/benchmark purposes.
  - We use only the `captions_val2017.json` annotation file, and download a small
    subset of individual images (default 200) directly by their `coco_url` field
    rather than the full ~1GB val2017.zip — this avoids downloading anything not
    used, keeps disk/bandwidth usage proportional to the benchmark size, and is
    fully scriptable/reproducible from a stable public URL with no authentication.
  - The subset (images + captions we actually use) is written under
    evaluation/datasets/coco_subset/ and is .gitignore'd (not committed) — rerun
    this script to reproduce it.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
OUTPUT_DIR = Path(__file__).parent / "coco_subset"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def ensure_annotations() -> Path:
    """Downloads only captions_val2017.json (extracted from the annotations zip,
    cached after first run)."""
    ann_path = OUTPUT_DIR / "captions_val2017.json"
    if ann_path.exists():
        return ann_path

    zip_path = OUTPUT_DIR / "annotations_trainval2017.zip"
    print("Downloading COCO 2017 annotations (captions only needed, ~250MB zip)...")
    download_file(ANNOTATIONS_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("annotations/captions_val2017.json") as src, open(ann_path, "wb") as dst:
            dst.write(src.read())
    zip_path.unlink()
    return ann_path


def build_subset(n_images: int, seed: int) -> dict:
    ann_path = ensure_annotations()
    with open(ann_path) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    captions_by_image: dict[int, list[str]] = {}
    for ann in coco["annotations"]:
        captions_by_image.setdefault(ann["image_id"], []).append(ann["caption"])

    import random

    rng = random.Random(seed)
    image_ids = [i for i in images_by_id if i in captions_by_image]
    rng.shuffle(image_ids)
    selected_ids = image_ids[:n_images]

    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for image_id in tqdm(selected_ids, desc="Downloading images"):
        img = images_by_id[image_id]
        dest = images_dir / img["file_name"]
        try:
            download_file(img["coco_url"], dest)
        except requests.exceptions.RequestException as exc:
            print(f"Skipping image {image_id} ({exc})")
            continue
        manifest.append({
            "image_id": image_id,
            "file_name": img["file_name"],
            "image_path": str(dest),
            "captions": captions_by_image[image_id],
        })

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Subset ready: {len(manifest)} images, manifest at {manifest_path}")
    return {"manifest_path": str(manifest_path), "n_images": len(manifest)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_subset(args.n_images, args.seed)
