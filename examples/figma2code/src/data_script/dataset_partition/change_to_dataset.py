import json
from pathlib import Path
from datasets import Dataset, Features, Value, Image, Sequence
from PIL import Image as PILImage

def load_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def extract_report_subset(report: dict):
    return {
        "file_key": report.get("file_key"),
        "node_id": report.get("node_id"),
        "page_url": report.get("page_url"),
        "annotation": report.get("annotation", {}),
        "statistics": report.get("statistics", {}),
    }

def pil_image_or_none(path: Path):
    if not path.exists():
        return None
    img = PILImage.open(path)
    if img.mode == "P" or img.mode == "LA":
        img = img.convert("RGBA")
    elif img.mode not in ["RGB", "RGBA"]:
        img = img.convert("RGB")
    return img

def build_dataset(base_dir: str, output_path: str):
    base_dir = Path(base_dir)

    features = Features({
        "root": Image(),              # PIL.Image
        "filekey": Value("string"),
        "node_id": Value("string"),
        "page_url": Value("string"),
        "annotation": Value("string"),
        "statistics": Value("string"),
        "image_refs": Sequence(Image()),  # PIL.Image list
        "svg_assets": Sequence(Value("string"))
    })

    samples = []

    for filekey_dir in base_dir.iterdir():
        if not filekey_dir.is_dir():
            continue

        report = load_json(filekey_dir / "report.json")
        report_subset = extract_report_subset(report)

        # root.png
        root_path = filekey_dir / "root.png"
        root_img = pil_image_or_none(root_path)

        # image_refs
        image_refs_dir = filekey_dir / "assets" / "image_refs"
        image_refs_imgs = []
        if image_refs_dir.exists():
            for p in sorted(image_refs_dir.glob("*")):
                img = pil_image_or_none(p)
                if img is not None:
                    image_refs_imgs.append(img)

        # svg_assets: directly take the value of report["assets"]["svg_assets"] and deduplicate
        svg_assets_map = report.get("downloaded_resources", {}).get("assets/svg_assets", {})
        svg_assets = set(svg_assets_map.values())  # Deduplicate


        sample = {
            "root": root_img,
            "filekey": report_subset.get("file_key"),
            "node_id": report_subset.get("node_id"),
            "page_url": report_subset.get("page_url"),
            "annotation": json.dumps(report_subset.get("annotation", {}), ensure_ascii=False),
            "statistics": json.dumps(report_subset.get("statistics", {}), ensure_ascii=False),
            "image_refs": image_refs_imgs,
            "svg_assets": list(svg_assets),
        }

        samples.append(sample)

    ds = Dataset.from_list(samples, features=features)
    ds.to_parquet(output_path)
    print(f"✅ Saved dataset to {output_path}, total {len(ds)} samples")

if __name__ == "__main__":
    from ...configs.paths import DATA_DIR
    build_dataset(
        str(DATA_DIR / "data_test"),
        str(DATA_DIR / "datasets" / "all_samples.parquet")
    )
