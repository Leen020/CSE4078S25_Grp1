import os
import pandas as pd
import json
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict

input_dir = "delivery1"
output_dir = "delivery2"

# Normalize entity names
entity_map = {
    "per": "PERSON",
    "person": "PERSON",
    "loc": "LOCATION",
    "location": "LOCATION",
    "org": "ORGANIZATION",
    "organization": "ORGANIZATION",
    "misc": "MISC"
}

def normalize_tag(tag, prev_entity="O"):
    tag = tag.strip()
    if tag.upper() == "O" or tag == "":
        return "O"

    parts = tag.split("-")

    if len(parts) == 2 and parts[0] in ("B", "I"):
        prefix, ent = parts
        ent_norm = entity_map.get(ent.lower(), ent.upper())
        return f"{prefix}-{ent_norm}"
    else:
        ent_norm = entity_map.get(tag.lower(), tag.upper())
        prefix = "B" if prev_entity != ent_norm else "I"
        return f"{prefix}-{ent_norm}"

def process_tags(tags):
    normalized = []
    prev_ent = "O"
    for tag in tags:
        new_tag = normalize_tag(tag, prev_ent)
        normalized.append(new_tag)
        prev_ent = new_tag.split("-")[1] if "-" in new_tag else "O"
    return normalized

def process_file(filepath, output_path):
    ext = filepath.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(filepath, encoding="utf-8", errors="replace")
        if "tag" in df.columns:
            df["tag"] = process_tags(df["tag"])
        elif "label" in df.columns:
            df["label"] = process_tags(df["label"])
        df.to_csv(output_path, index=False, encoding="utf-8")

    elif ext == ".tsv":
        df = pd.read_csv(filepath, sep="\t", encoding="utf-8")
        if "Tag" in df.columns:
            df["Tag"] = process_tags(df["Tag"])
        elif "Word" in df.columns:
            df["Word"] = process_tags(df["Word"])
        df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")

    elif ext == ".json":
        with filepath.open("r", encoding="utf-8", errors="replace") as f:
            try:
                data = json.load(f)
                is_json_lines = False
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]
                is_json_lines = True

        for entry in data:
            if "tag" in entry:
                entry["tag"] = process_tags(entry["tag"])
            elif "label" in entry:
                entry["label"] = process_tags(entry["label"])
            elif "spans" in entry:
                for span in entry["spans"]:
                    ent_type = span.get("label", "")
                    ent_norm = entity_map.get(ent_type.lower(), ent_type.upper())
                    span["label"] = ent_norm

        with output_path.open("w", encoding="utf-8") as f:
            if is_json_lines:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)

    elif ext == ".parquet":
        table = pq.read_table(filepath)
        df = table.to_pandas()
        if "tag" in df.columns:
            df["tag"] = process_tags(df["tag"])
        elif "label" in df.columns:
            df["label"] = process_tags(df["label"])
        df.to_parquet(output_path, index=False)

    elif ext in [".txt", ".conll"]:
        lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
        new_lines = []
        prev_entity = "O"
        for line in lines:
            if line.strip() == "":
                new_lines.append("")
                prev_entity = "O"
                continue
            parts = line.split()
            tag = parts[-1]
            new_tag = normalize_tag(tag, prev_entity)
            new_lines.append(" ".join(parts[:-1] + [new_tag]))
            prev_entity = new_tag.split("-")[1] if "-" in new_tag else "O"
        output_path.write_text("\n".join(new_lines), encoding="utf-8")

def main():
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    for file in input_path.rglob("*.*"):
        if file.suffix.lower() not in [".csv", ".json", ".conll", ".parquet", ".tsv", ".txt"]:
            continue
        relative_path = file.relative_to(input_path)
        out_file = output_path / relative_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            process_file(file, out_file)
            print(f"✔ Processed {relative_path}")
        except Exception as e:
            print(f"❌ Error in {relative_path}: {e}")

if __name__ == "__main__":
    main()

# training_dataset_paths, test_dataset_paths, dev_dataset_paths = split_train_test_valid_datasets(paths)



