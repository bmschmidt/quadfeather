import argparse
from pathlib import Path
import pyarrow as pa
from pyarrow import compute as pc
from pyarrow import feather
import json
from typing import Iterator


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
        help="key to use for joining; must exist in both tables",
        required=True,
    )
    return parser.parse_args()


class Tileset:
    def __init__(self, dir: Path):
        self.dir = dir

    def __iter__(self):
        """
        Iterate over all the base level files in the directory.
        """
        for file in self.dir.glob("*/*/*.feather"):
            if len(file.stem.split(".")) > 1:
                # Ignore sidecar files
                continue
            yield file

    def sidecar_iter(self, sidecar):
        """
        Iterate over all sidecar files in the directory.
        """
        for file in self.dir.glob("*/*/*.feather"):
            splat = file.stem.split(".")
            if len(splat) == 1:
                continue
            if splat[1] != sidecar:
                continue
            yield file

    def add_sidecars(self, sidecar: Path, key: str, filename=None):
        """
        Add sidecar files to a tileset.

        Parameters
        ----------
        sidecar: Path A single feather file to add to the tileset.
        key: str The column to use for joining the sidecar to the tileset.
        filename: bool If present, the name of the file to write the sidecar to.
                  Otherwise, each column will get its own file named by the column name.
        """
        sidecar_table = feather.read_table(sidecar)
        master_lookup = []
        slices: list[tuple[int, int, Path]] = []
        start = 0
        for i, file in enumerate(self):
            table = feather.read_table(file, columns=[key])
            # table = table.append_column('file', pa.array([file.name] * len(table)))
            master_lookup.append(table)
            slices.append((start, start + len(table), file))
            start += len(table)
            if len(table) > 66_000:
                raise ValueError(
                    f"file {file} has more than 66k rows; this indicates a problem with the tileset"
                )
        master_lookup_tb = pa.concat_tables(master_lookup)
        all_ixes = pc.index_in(master_lookup_tb[key], sidecar_table[key])

        locations = {}
        for start, end, file in slices:
            print(file)
            # end-start because for pyarrow slice takes a length, not two offsets. Learned that the hard way.
            ixes = all_ixes.slice(start, end - start)
            tb = sidecar_table.take(ixes).combine_chunks()
            assert len(tb.to_batches()) == 1
            if filename is None:
                for column in tb.column_names:
                    if column != key:
                        locations[column] = column
                        fout = file.with_suffix(f".{column}.feather")
                        feather.write_feather(
                            tb.select([column]), fout, compression="uncompressed"
                        )
            else:
                fout = file.with_suffix(f".{filename}.feather")
                locations = {
                    key: filename for column in tb.column_names if column != key
                }
                feather.write_feather(tb.drop(key), fout, compression="uncompressed")

        # Overwrite the root tile with information about the sidecars.
        rootpath = self.dir / "0/0/0.feather"
        root = feather.read_table(rootpath)
        locations = {
            column: column for column in sidecar_table.column_names if column != key
        }
        # Copy the existing sidecar locations, if any.
        locations = {
            **json.loads(root.schema.metadata.get(b"sidecars", b"{}")),
            **locations,
        }
        root = root.replace_schema_metadata(
            {**root.schema.metadata, b"sidecars": json.dumps(locations).encode("utf-8")}
        )
        feather.write_feather(root, rootpath, compression="uncompressed")


def add_sidecars_cli():
    args = parse_args()
    tileset = Tileset(args.tileset)
    tileset.add_sidecars(args.sidecar, args.key)


if __name__ == "__main__":
    add_sidecars_cli()
