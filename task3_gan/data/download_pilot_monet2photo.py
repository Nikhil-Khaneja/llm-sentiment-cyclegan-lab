"""PILOT DATA ONLY - not the class competition data.

Downloads the public CycleGAN-paper monet2photo set (huggan/monet2photo on Hugging Face) into
task3_gan/data/pilot_monet2photo/{monet_jpg,photo_jpg}. Used only by configs/quick.yaml to
check training behaviour before the Kaggle data is available. Results from it are not
reportable and must never be submitted to Kaggle.

    python task3_gan/data/download_pilot_monet2photo.py
"""
import hashlib
import io
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image

OUT = Path(__file__).resolve().parent / "pilot_monet2photo"


def main():
    # Rows hold one image per domain as raw bytes: imageA = Monet, imageB = photo (CycleGAN
    # convention). The shorter domain repeats across rows, so de-duplicate by content hash.
    ds = load_dataset("huggan/monet2photo", split="train")
    seen, n = set(), {"monet_jpg": 0, "photo_jpg": 0}
    for d in n:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    for row in ds:
        for key, dom in [("imageA", "monet_jpg"), ("imageB", "photo_jpg")]:
            img = row.get(key)
            if not img or not img.get("bytes"):
                continue
            h = hashlib.sha1(img["bytes"]).hexdigest()
            if (dom, h) in seen:
                continue
            seen.add((dom, h))
            name = f"{dom.split('_')[0]}_{n[dom]:05d}"
            Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(OUT / dom / f"{name}.jpg", quality=95)
            n[dom] += 1
    print(f"wrote {n} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
