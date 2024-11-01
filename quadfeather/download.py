from pathlib import Path
import io
import requests
from pyarrow import ipc, feather
import sys
import argparse
import gzip
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a tileset from the quadfeather server."
    )

    # Positional arguments
    ## URL
    parser.add_argument(
        "url", metavar="url", type=str, help="URL of the tileset to download."
    )
    ## Destination folder
    parser.add_argument(
        "dest",
        metavar="dest",
        type=Path,
        help="Destination folder for the downloaded tileset.",
    )
    return parser.parse_args()


def download_tile(url, tile, dest, suffix=".feather"):
    if not url.endswith("/"):
        url = url + "/"
    source = url + tile + suffix
    resp = requests.get(str(source))
    if resp.status_code != 200:
        if suffix == ".feather":
            return download_tile(url, tile, suffix=".feather.gz")
        else:
            raise FileNotFoundError(f"Error downloading {source} : {resp.status_code}")
    buffer = io.BytesIO(resp.content)
    if suffix == ".feather.gz":
        buffer = gzip.decompress(buffer.read())
        buffer = io.BytesIO(buffer)
    table = feather.read_table(buffer)
    file = Path(dest, tile).with_suffix(".feather")
    file.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(table, file, compression="uncompressed")
    print(file)
    if b"children" in table.schema.metadata:
        children = json.loads(table.schema.metadata[b"children"])
        for child in children:
            download_tile(url, child, dest, suffix=suffix)


def download_tileset(url, dest):
    if not dest.exists():
        dest.mkdir(parents=True)
    download_tile(url, tile="0/0/0", dest=dest, suffix=".feather")


if __name__ == "__main__":
    args = parse_args()
    print(args.url, args.dest)
    download_tileset(args.url, args.dest)
