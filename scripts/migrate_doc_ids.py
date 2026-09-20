"""Prefix legacy doc_ids and chunk_ids with the English source name."""

import json
from pathlib import Path

PREFIX = "kubernetes:"
FILES = [
    Path("data/evalsets/questions_raw.json"),
    Path("data/evalsets/questions_labelled.json"),
]


def migrate(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = 0

    for question in data["questions"]:
        doc_id = question["source_doc_id"]
        if not doc_id.startswith(PREFIX):
            question["source_doc_id"] = PREFIX + doc_id
            changed += 1

        gold = question.get("gold_chunk_ids")
        if gold:
            question["gold_chunk_ids"] = [
                cid if cid.startswith(PREFIX) else PREFIX + cid for cid in gold
            ]

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{path.name}: migrated {changed} questions")


for file in FILES:
    if file.is_file():
        migrate(file)
