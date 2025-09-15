#!/usr/bin/env python3
"""
update_vector_db.py
- For each generated sample, extract metadata + generate embeddings and upsert into vector DB
- Include 'score' and 'sha' as metadata so RAG can prefer high-scoring designs
"""
import argparse, json, os
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--score", required=True)
    args = p.parse_args()

    d = Path(args.dir)
    meta_file = d / "metadata.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
    else:
        meta = {}

    # TODO: generate embedding from SCAD text or summary and upsert into vector DB
    # Placeholder: print what would be upserted
    entry = {
        "id": meta.get("sha", "local"),
        "prompt": meta.get("prompt"),
        "score": float(args.score),
        "scad": (d / "model.scad").read_text() if (d / "model.scad").exists() else None
    }
    print("Would upsert to vector DB:", json.dumps(entry)[:1000])

if __name__ == "__main__":
    main()
