"""Download and extract the HetRec 2011 Last.fm-2k dataset."""

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import Config


def download_file(url: str, dest: Path) -> None:
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def main():
    cfg = Config()
    cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)

    marker = cfg.raw_data_dir / "user_artists.dat"
    if marker.exists():
        print(f"Dataset already exists at {cfg.raw_data_dir}, skipping download.")
        return

    zip_path = cfg.raw_data_dir / "hetrec2011-lastfm-2k.zip"
    print(f"Downloading HetRec 2011 Last.fm-2k from {cfg.dataset_url}...")
    download_file(cfg.dataset_url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            member_path = (cfg.raw_data_dir / member).resolve()
            if not str(member_path).startswith(str(cfg.raw_data_dir.resolve())):
                raise ValueError(f"Zip contains path traversal: {member}")
        zf.extractall(cfg.raw_data_dir)
    zip_path.unlink()

    expected_files = ["artists.dat", "tags.dat", "user_artists.dat", "user_taggedartists.dat"]
    for fname in expected_files:
        path = cfg.raw_data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found after extraction: {path}")

    print(f"Done. Files extracted to {cfg.raw_data_dir}")
    print(f"  artists.dat: {sum(1 for _ in open(cfg.raw_data_dir / 'artists.dat', encoding='latin-1')) - 1} artists")
    print(f"  user_artists.dat: {sum(1 for _ in open(cfg.raw_data_dir / 'user_artists.dat', encoding='latin-1')) - 1} interactions")
    print(f"  tags.dat: {sum(1 for _ in open(cfg.raw_data_dir / 'tags.dat', encoding='latin-1')) - 1} tags")


if __name__ == "__main__":
    main()
