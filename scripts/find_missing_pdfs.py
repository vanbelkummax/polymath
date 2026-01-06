#!/usr/bin/env python3
"""Find PDFs that aren't in ChromaDB yet"""

import chromadb
from pathlib import Path

CHROMADB_PATH = "/home/user/work/polymax/chromadb/polymath_v2"

FOLDERS = [
    "/mnt/c/Users/User/Downloads/Newpapers6",
    "/mnt/c/Users/User/Downloads/Newpapers5",
    "/mnt/c/Users/User/Downloads/Newpapers 4",
    "/mnt/c/Users/User/Downloads/Newpapers 3",
    "/mnt/c/Users/User/Downloads/Newpapers2",
    "/mnt/c/Users/User/Downloads/Paperfulltext",
]

# Get all sources in ChromaDB
print("Loading ChromaDB...")
client = chromadb.PersistentClient(path=CHROMADB_PATH)
coll = client.get_collection("polymath_corpus")

# Get unique sources (PDF filenames)
result = coll.get(include=["metadatas"])
existing_sources = set()
for meta in result['metadatas']:
    if meta and 'source' in meta:
        existing_sources.add(meta['source'])

print(f"Found {len(existing_sources)} unique sources in ChromaDB")

# Find missing PDFs
missing = []
for folder in FOLDERS:
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"Folder not found: {folder}")
        continue

    for pdf in folder_path.glob("*.pdf"):
        if pdf.name not in existing_sources:
            missing.append(str(pdf))

print(f"\n{len(missing)} PDFs not in ChromaDB:")
for pdf in missing[:30]:
    print(f"  {Path(pdf).name}")
if len(missing) > 30:
    print(f"  ... and {len(missing) - 30} more")

# Write to file for batch processing
with open("/home/user/work/polymax/missing_pdfs.txt", "w") as f:
    for pdf in missing:
        f.write(pdf + "\n")

print(f"\nFull list saved to: /home/user/work/polymax/missing_pdfs.txt")
