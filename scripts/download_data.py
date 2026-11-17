"""Local inventory only: automatic network downloads disabled by user request."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'data/raw'
for path in sorted(root.rglob('*')):
    if path.is_file(): print(path.relative_to(root),path.stat().st_size)
