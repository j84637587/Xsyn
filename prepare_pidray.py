"""Convert PIDray COCO-format annotation to the .pt format expected by Caption dataset.

Usage:
    python prepare_pidray.py \
        --ann_file data/pidray/annotations/instances_train.json \
        --output    data/pidray/pidray_train.pt
"""
import argparse
import json
import torch
from pathlib import Path


# PIDray category id → name mapping (matches gligen_inference.py)
ID2NAME = {
    1: "Baton", 2: "Pliers", 3: "Hammer", 4: "Powerbank",
    5: "Scissors", 6: "Wrench", 7: "Gun", 8: "Bullet",
    9: "Sprayer", 10: "HandCuffs", 11: "Knife", 12: "Lighter",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file", required=True, help="path to instances_train.json")
    parser.add_argument("--output",   required=True, help="output .pt path")
    args = parser.parse_args()

    with open(args.ann_file, "r") as f:
        coco = json.load(f)

    # Build lookup tables
    id2filename = {img["id"]: img["file_name"] for img in coco["images"]}
    id2catname  = {cat["id"]: cat["name"] for cat in coco["categories"]}
    # Fall back to our own mapping if dataset uses numeric-only names
    for cid, name in ID2NAME.items():
        id2catname.setdefault(cid, name)

    # Group annotations by image
    img2annos: dict = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        img2annos.setdefault(img_id, []).append(ann)

    data_list = []
    for img_id, filename in id2filename.items():
        annos = img2annos.get(img_id, [])
        if not annos:
            continue

        processed_annos = []
        category_names = []
        for ann in annos:
            x, y, w, h = ann["bbox"]          # COCO format: x,y,w,h
            x1, y1, x2, y2 = x, y, x+w, y+h  # convert to x1,y1,x2,y2
            cat_name = id2catname.get(ann["category_id"], "unknown")
            processed_annos.append({
                "bbox":    [x1, y1, x2, y2],
                "caption": f"one {cat_name}",
            })
            category_names.append(cat_name)

        caption = ", ".join(category_names)

        data_list.append({
            "file_path": Path(filename).name,  # just the filename, no subdir prefix
            "captions":  caption,
            "annos":     processed_annos,
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_list, args.output)
    print(f"Saved {len(data_list)} entries → {args.output}")


if __name__ == "__main__":
    main()
