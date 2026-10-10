"""下载 Stanford 官方 GloVe 6B，并仅解压 100 维词向量文件。"""

import argparse
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path


GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_NAME = "glove.6B.100d.txt"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "resources" / "glove"


def download_glove(directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / GLOVE_NAME
    if target.is_file():
        print(f"Using existing file: {target.resolve()}", flush=True)
        return target

    archive_path = directory / "glove.6B.zip"
    if not archive_path.is_file():
        partial = directory / "glove.6B.zip.part"
        request = urllib.request.Request(GLOVE_URL, headers={"User-Agent": "Mozilla/5.0"})
        print(f"Downloading official GloVe: {GLOVE_URL}", flush=True)
        started = last_log = time.perf_counter()
        downloaded = 0
        with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as file:
            total = int(response.headers.get("Content-Length", 0))
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
                downloaded += len(chunk)
                now = time.perf_counter()
                if now - last_log >= 10:
                    total_text = f" / {total / 1024**2:.1f}" if total else ""
                    print(
                        f"  {downloaded / 1024**2:.1f}{total_text} MiB, "
                        f"{downloaded / 1024**2 / (now - started):.1f} MiB/s",
                        flush=True,
                    )
                    last_log = now
        if total and downloaded != total:
            raise RuntimeError(f"下载不完整：预期 {total} 字节，实际 {downloaded} 字节。")
        # 只有完整且包含目标文件的 ZIP 才能成为正式下载文件。
        with zipfile.ZipFile(partial) as archive:
            archive.getinfo(GLOVE_NAME)
        partial.replace(archive_path)

    print(f"Extracting {GLOVE_NAME} ...", flush=True)
    partial_text = directory / (GLOVE_NAME + ".part")
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(GLOVE_NAME)
        with archive.open(info) as source, partial_text.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    if partial_text.stat().st_size != info.file_size:
        raise RuntimeError("解压文件大小不正确。")
    partial_text.replace(target)
    print(f"Ready: {target.resolve()}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    download_glove(args.directory)


if __name__ == "__main__":
    main()
