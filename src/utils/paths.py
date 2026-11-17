from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def project_path(path):
    return ROOT / path
