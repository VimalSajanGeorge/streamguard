# cleanup_sard.py — run from the sard root directory
import shutil
from pathlib import Path

KEEP_PREFIXES = (
    "CWE78",
    "CWE121",
    "CWE122",
    "CWE134",
    "CWE190",
    "CWE416",
    "CWE476",
)

sard_root = Path("C:/Users/Vimal Sajan/streamguard/data/raw/raw/sard")  # adjust to your actual path

deleted = 0
kept = 0
for entry in sard_root.iterdir():
    if not entry.is_dir():
        continue
    if entry.name.startswith(KEEP_PREFIXES):
        kept += 1
    else:
        shutil.rmtree(entry)
        deleted += 1

print(f"Kept: {kept} directories")
print(f"Deleted: {deleted} directories")