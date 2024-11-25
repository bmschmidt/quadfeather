import argparse
from pathlib import Path
import pyarrow as pa
from pyarrow import compute as pc
from .tiler import Quadtree
from .ingester import get_ingester


def parse_args():
    parser = argparse.ArgumentParser(description="Add sidecar files to a tileset.")
    parser.add_argument(
        "--tileset",
        type=Path,
        required=True,
        help="Path to the tileset to add sidecars to.",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        required=True,
        help="Path to the new data to add to the tileset.",
    )
    parser.add_argument(
        "--key",
        type=str,
        help="key to use for joining; must exist in both tables with the same name",
        required=True,
    )
    parser.add_argument(
        "--sidecar-name",
        type=str,
        default=None,
        help="Name of the new ",
        required=False,
    )
    return parser.parse_args()


def add_sidecars_cli():
    args = parse_args()
    tileset = Quadtree.from_dir(args.tileset, mode="append")

    ingester = get_ingester(args.sidecar, False)

    tileset.join(
        data=ingester.batches(), id_field=args.key, new_sidecar_name=args.sidecar_name
    )


if __name__ == "__main__":
    add_sidecars_cli()
