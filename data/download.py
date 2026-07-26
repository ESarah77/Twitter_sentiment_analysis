from pathlib import Path
import subprocess
import zipfile

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

subprocess.run(
    [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "jp797498e/twitter-entity-sentiment-analysis",
        "-p",
        str(DATA_DIR),
    ],
    check=True,
)

with zipfile.ZipFile(next(DATA_DIR.glob("*.zip"))) as z:
    z.extractall(DATA_DIR)