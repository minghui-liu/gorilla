import json
from pathlib import Path

TRAIN_IDS_PATH = Path("BFCL_v4_multi_turn_base_train.json")
INPUT_PATH = Path("Qwen3-8B.json")
OUTPUT_PATH = Path("Qwen3-8B.train.json")


def load_train_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.add(obj["id"])
    return ids


def main() -> None:
    train_ids = load_train_ids(TRAIN_IDS_PATH)

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = [row for row in data if row.get("task_name") in train_ids]

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Input rows: {len(data)}")
    print(f"Filtered rows: {len(filtered)}")
    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
