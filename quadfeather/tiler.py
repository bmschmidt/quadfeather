import pyarrow as pa
from pyarrow import csv, feather, parquet as pq, compute as pc, ipc as ipc

import logging
from pathlib import Path
import sys
import argparse
from io import BytesIO
import json
import hashlib
from base64 import urlsafe_b64encode
from numpy import random as nprandom
from collections import defaultdict, Counter
from typing import (
    Iterator,
    Dict,
    List,
    Tuple,
    Optional,
    Any,
    Literal,
    Union,
    Tuple,
)
import numpy as np
import sys
from math import isfinite, sqrt
from .ingester import get_ingester
from dataclasses import dataclass

logger = logging.getLogger("quadfeather")
logger.setLevel(logging.DEBUG)
DEFAULTS: Dict[str, Any] = {
    "first_tile_size": 1000,
    "tile_size": 50000,
    "destination": Path("."),
    "max_open_filehandles": 33.0,
    "randomize": 0.0,
    "files": None,
    "limits": [float("inf"), float("inf"), -float("inf"), -float("inf")],
    "log_level": 30,
}


@dataclass
class Rectangle:
    x: Tuple[float, float]
    y: Tuple[float, float]


@dataclass
class TileManifest:
    key: str
    nPoints: int
    children: List["TileManifest"]
    min_ix: int
    max_ix: int
    extent: Rectangle

    def to_dict(self):
        d = {**self.__dict__}
        d["children"] = [child.to_dict() for child in self.children]
        d["extent"] = {"x": list(self.extent.x), "y": list(self.extent.y)}
        return d


def flatten_manifest(mani: TileManifest) -> List[Dict]:
    # Flatten a manifest into a table.
    # This is a recursive function.
    d = [
        {
            "key": mani.key,
            "nPoints": mani.nPoints,
            "min_ix": mani.min_ix,
            "max_ix": mani.max_ix,
            "extent": json.dumps(mani.to_dict()["extent"]),
        }
    ]
    for child in mani.children:
        d += flatten_manifest(child)
    return d


def rebatch(
    input: Iterator[pa.RecordBatch], size: float, max_rows: float = float("inf")
) -> Iterator[pa.Table]:
    """
    Assembles an iterator over record batches into a predictable memory size

    Yields *more* than the requested size to avoid bad situations of zero-size arrays.
    """
    buffer = []
    buffer_size = 0
    buffer_rows = 0
    for batch in input:
        buffer_size += batch.nbytes
        buffer_rows += batch.num_rows
        buffer.append(batch)

        if buffer_size > size or buffer_rows > max_rows:
            tb = pa.Table.from_batches(buffer).combine_chunks()
            buffer = []
            buffer_size = 0
            buffer_rows = 0
            yield tb
    if len(buffer) > 0:
        tb = pa.Table.from_batches(buffer)
        yield tb.combine_chunks()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tile an input file into a large number of arrow files."
    )
    parser.add_argument(
        "--first_tile_size",
        type=int,
        default=DEFAULTS["first_tile_size"],
        help="Number of records in first tile.",
    )

    parser.add_argument(
        "--tile_size",
        type=int,
        default=DEFAULTS["tile_size"],
        help="Number of records per tile.",
    )
    parser.add_argument(
        "--destination",
        "--directory",
        "-d",
        type=Path,
        required=True,
        help="Destination directory to write to.",
    )
    parser.add_argument(
        "--max-open-filehandles",
        type=float,
        default=DEFAULTS["max_open_filehandles"],
        help="Max files to have open at once. Default 25. "
        "Usually it's much faster to have significantly fewer files than the max allowed open by the system ",
    )
    parser.add_argument(
        "--randomize",
        type=float,
        default=DEFAULTS["randomize"],
        help="Uniform random noise to add to points. If you have millions "
        "of coincident points, can reduce the depth of the tree greatly.",
    )

    parser.add_argument(
        "--files",
        "-f",
        nargs="+",
        type=Path,
        help="""Input file(s). If column 'ix' defined, will be used: otherwise, indexes will be assigned. Must be in sorted order.""",
    )

    parser.add_argument(
        "--limits",
        nargs=4,
        type=float,
        metavar=list,
        default=[float("inf"), float("inf"), -float("inf"), -float("inf")],
        help="""Data limits, in [x0 y0 xmax ymax] order. If not entered, will be calculated at some cost.""",
    )

    parser.add_argument("--log-level", type=int, default=30)
    arguments = sys.argv[1:]
    args = parser.parse_args(arguments)

    return {**DEFAULTS, **vars(args)}


# First, we guess at the schema using pyarrow's own type hints.
# This could be overridden by user args.
def refine_schema(schema: pa.Schema) -> Dict[str, pa.DataType]:
    """ " """
    fields = {}
    seen_ix = False
    for el in schema:
        if el.name == "ix":
            fields[el.name] = pa.uint64()
            seen_ix = True
        elif pa.types.is_date(el.type):
            # Deepscatter can't handle all the different date types.
            fields[el.name] = pa.timestamp("ms")
        else:
            fields[el.name] = el.type
    if not seen_ix:
        fields["ix"] = pa.uint64()
    return fields


def determine_schema(files: List[Path]):
    vals = csv.open_csv(
        files[0],
        csv.ReadOptions(block_size=1024 * 1024 * 64),
        convert_options=csv.ConvertOptions(
            auto_dict_encode=True, auto_dict_max_cardinality=4094
        ),
    )
    override = {}

    raw_schema = vals.read_next_batch().schema

    schema = {}
    for el in raw_schema:
        t = el.type
        schema[el.name] = t
        if el.name in override:
            schema[el.name] = getattr(pa, override[el.name])()
    schema["ix"] = pa.uint64()
    schema_safe = dict(
        [
            (k, v if not pa.types.is_dictionary(v) else pa.string())
            for k, v in schema.items()
        ]
    )
    return schema, schema_safe


# Next, convert these CSVs into some preliminary arrow files.
# These are the things that we'll actually read in.

# We write to arrow because we need a first pass anyway to determine
# the data bounds and some other stuff; and it will be much faster to re-parse
# everything from arrow than from CSV.


def rewrite_in_arrow_format(
    files,
    schema_safe: pa.Schema,
    schema: pa.Schema,
    csv_block_size: int = 1024 * 1024 * 128,
):
    # Returns: an extent and a list of feather files.
    # files: a list of csvs
    # schema_safe: the output schema (with no dictionary types)
    # schema: the input schema (with dictionary types)
    # csv_block_size: the block size to use when reading the csv files.

    # ix = 0
    extent = Rectangle(
        x=(float("inf"), -float("inf")),
        y=(float("inf"), -float("inf")),
    )
    output_dir = files[0].parent / "_deepscatter_tmp"
    output_dir.mkdir()
    # if "z" in schema.keys():
    #     extent["z"] = [float("inf"), -float("inf")]

    rewritten_files: list[Path] = []
    for FIN in files:
        vals = csv.open_csv(
            FIN,
            csv.ReadOptions(block_size=csv_block_size),
            convert_options=csv.ConvertOptions(column_types=schema_safe),
        )
        for chunk_num, batch in enumerate(vals):
            # Loop through the whole CSV, writing out 100 MB at a time,
            # and converting each batch to dictionary as we go.
            d = dict()
            for i, name in enumerate(batch.schema.names):
                if pa.types.is_dictionary(schema[name]):  ###
                    d[name] = batch[i].dictionary_encode()  ###
                else:  ###
                    d[name] = batch[i]  ###
            data = pa.Table.from_batches([batch])
            # Count each element in a uint64 array (float32 risks overflow,
            # Uint32 supports up to 2 billion or so, which is cutting it close for stars.)
            # d["ix"] = pa.array(range(ix, ix + len(batch)), type=pa.uint64())
            # ix += len(batch)
            xlim = pc.min_max(data["x"]).as_py()
            ylim = pc.min_max(data["y"]).as_py()
            extent.x = (min(extent.x[0], xlim["min"]), max(extent.x[1], xlim["max"]))
            extent.y = (min(extent.y[0], ylim["min"]), max(extent.y[1], ylim["max"]))
            final_table = pa.table(d)
            fout = output_dir / f"{chunk_num}.feather"
            feather.write_feather(final_table, fout, compression="zstd")
            rewritten_files.append(fout)
    raw_schema = final_table.schema
    return rewritten_files, extent, raw_schema
    # Learn the schema from the last file written out.


def check_filesnames(files: List[Path]):
    ftypes = [f.suffix for f in files]
    for f in ftypes:
        if f != f[0]:
            raise TypeError(
                f"Must pass all the same type of file as input, not {f}/{f[0]}"
            )
    if not f in set(["arrow", "feather", "csv", "gz"]):
        raise TypeError("Must use files ending in 'feather', 'arrow', or '.csv'")


def main(
    files: Union[List[str], List[Path]],
    destination: Union[str, Path],
    extent: Union[None, Rectangle, Dict[str, Tuple[float, float]]] = None,
    csv_block_size=1024 * 1024 * 128,
    tile_size=65_000,
    first_tile_size=2000,
    dictionaries: Dict[str, pa.Array] = {},
    sidecars: Dict[str, str] = {},
    # Actually dict of pa type constructors.
    schema: pa.Schema = pa.schema({}),
    max_open_filehandles=33,
    randomize=0,
    limits=None,
):
    """
    Run a tiler

    arguments: a list of strings to parse. If None, treat as command line args.
    csv_block_size: the block size to use when reading the csv files. The default should be fine,
        but included here to allow for testing multiple blocks.

    """

    if extent is None and limits is not None:
        extent = Rectangle(x=(limits[0], limits[2]), y=(limits[1], limits[3]))

    files = [Path(file) for file in files]
    destination = Path(destination)
    if extent is not None and isinstance(extent, dict):
        extent = Rectangle(**extent)
    dirs_to_cleanup = []
    if files[0].suffix == ".csv" or str(files[0]).endswith(".csv.gz"):
        schema, schema_safe = determine_schema(files)
        # currently the dictionary type isn't supported while reading CSVs.
        # So we have to do some monkey business to store it as keys at first, then convert to dictionary later.
        logger.debug(schema)
        rewritten_files, extent, raw_schema = rewrite_in_arrow_format(
            files, schema_safe, schema, csv_block_size
        )
        dirs_to_cleanup.append(rewritten_files[0].parent)
        logger.info("Done with preliminary build")
    else:
        rewritten_files = files
        if extent is None or extent.x[0] == float("inf"):
            fin = files[0]
            if fin.suffix == ".feather" or fin.suffix == ".parquet":
                reader = get_ingester(files)
                xmin, xmax = (float("inf"), -float("inf"))
                ymin, ymax = (float("inf"), -float("inf"))
                for batch in reader:
                    x = pc.min_max(batch["x"]).as_py()
                    y = pc.min_max(batch["y"]).as_py()
                    if x["min"] < xmin and isfinite(x["min"]):
                        xmin = x["min"]
                    if x["max"] > xmax and isfinite(x["max"]):
                        xmax = x["max"]
                    if y["min"] < ymin and isfinite(y["min"]):
                        ymin = y["min"]
                    if y["max"] > ymax and isfinite(y["max"]):
                        ymax = y["max"]
                extent = Rectangle(x=(xmin, xmax), y=(ymin, ymax))
        if files[0].suffix == ".feather":
            first_batch = ipc.RecordBatchFileReader(files[0])
            raw_schema = first_batch.schema
        elif files[0].suffix == ".parquet":
            first_batch = pq.ParquetFile(files[0])
            raw_schema = first_batch.schema_arrow
        schema = refine_schema(raw_schema)
        elements = {name: type for name, type in schema.items()}
        if not "ix" in elements:
            # Must always be an ix field.
            elements["ix"] = pa.uint64()
        raw_schema = pa.schema(elements)

    logger.info("Starting.")

    recoders: Dict = dict()

    for field in raw_schema:
        if pa.types.is_dictionary(field.type):
            recoders[field.name] = get_recoding_arrays(rewritten_files, field.name)

    if extent is None:
        raise ValueError("Extent must be defined.")

    tiler = Quadtree(
        mode="write",
        schema=pa.schema(schema),
        extent=extent,
        basedir=destination,
        first_tile_size=first_tile_size,
        tile_size=tile_size,
        sidecars=sidecars,
        max_open_filehandles=max_open_filehandles,
        dictionaries=dictionaries,
        randomize=randomize,
    )
    tiler.insert_files(files=rewritten_files)

    logger.info("Job complete.")
    return tiler


def partition(table: pa.Table, midpoint: Tuple[str, float]) -> List[pa.Table]:
    # Divide a table in two based on a midpoint
    key, pivot = midpoint
    criterion = pc.less(table[key], np.float32(pivot))
    splitted = [pc.filter(table, criterion), pc.filter(table, pc.invert(criterion))]
    return splitted


class Quadtree:
    def __init__(
        self,
        basedir: Path,
        extent: Union[Rectangle, Tuple[Tuple[float, float], Tuple[float, float]]],
        mode: Literal["read", "write", "append"] = "read",
        dictionaries: Dict[str, pa.Array] = {},
        schema: pa.Schema = pa.schema({}),
        first_tile_size=2000,
        tile_size=65_000,
        max_open_filehandles=32,
        sidecars: Dict[str, str] = {},
        randomize=0,
    ):
        """
        * sidecars: a dictionary of sidecar files to add to the tileset. The key is the column that
            should be written in a sidecar, and the value is the extension for the sidecar file.
        """
        logger.debug("Creating quadtree")
        self.tile_size = tile_size
        self.first_tile_size = first_tile_size
        self.basedir = basedir
        self.dictionaries = dictionaries
        self.max_open_filehandles = max_open_filehandles
        self.mode = mode
        self.randomize = randomize
        # if mode != "write":
        #     raise NotImplementedError("Only write mode is supported right now.")
        self.sidecars = sidecars
        self._insert_schema = None
        if isinstance(extent, tuple):
            extent = Rectangle(x=extent[0], y=extent[1])
        self.schema = schema
        self._schemas: Optional[Dict[str, pa.Schema]] = None
        if self.mode == "write":
            self.root = Tile(
                quadtree=self,
                extent=extent,
                permitted_children=max_open_filehandles - 1,
                tile_code=(0, 0, 0),
            )
        self._tiles = None
        self._macrotiles: Optional[List[Macrotile]] = None
        self._bloom_cache: Dict[str, pa.Array] = {}

    @staticmethod
    def from_dir(basedir: Path, mode: Literal["read", "append"]) -> "Quadtree":
        # Load a quadtree from disk.
        manifest = basedir / "manifest.feather"
        if not manifest.exists():
            raise FileNotFoundError(
                "Not able to load a quadtree without a manifest file."
            )
        manifest = feather.read_table(basedir / "manifest.feather")
        sidecars = json.loads(manifest.schema.metadata[b"sidecars"])
        schema = pa.ipc.read_schema(BytesIO(bytes(manifest.schema.metadata[b"schema"])))
        extent = manifest.filter(pc.equal(manifest["key"], "0/0/0"))["extent"][
            0
        ].as_py()
        loaded = json.loads(extent)
        extent = Rectangle(
            x=tuple(loaded["x"]),
            y=tuple(loaded["y"]),
        )
        return Quadtree(
            basedir=basedir,
            extent=extent,
            mode=mode,
            sidecars=sidecars,
            schema=schema,
        )

    def tiles(self):
        if self._tiles is not None:
            return self._tiles
        self._tiles = [*self.iterate()]
        return self._tiles

    @property
    def macrotiles(self):
        if self._macrotiles is not None:
            return self._macrotiles
        self._macrotiles = [*self.iter_macrotiles()]
        return self._macrotiles

    @property
    def schemas(self) -> Dict[str, pa.Schema]:
        if self._schemas is not None:
            return self._schemas
        fields = defaultdict(list)
        assert self._insert_schema is not None
        for field in self._insert_schema:
            car = self.sidecars.get(field.name, "")
            dtype = field.type
            if field.name in self.schema.names:
                # Check if the user requested a cast
                dtype = self.schema.field(field.name).type
            if field in self.dictionaries:
                # dictionaries are written later.
                dtype = pa.dictionary(pa.int16(), pa.utf8())
            fields[car].append(pa.field(field.name, dtype))

        fields[self.sidecars.get("ix", "")].append(pa.field("ix", pa.uint64()))
        self._schemas = {k: pa.schema(v) for k, v in fields.items()}
        return self._schemas

    def insert(self, tab):
        """
        Insert's a table.
        """
        if self._insert_schema is None:
            self._insert_schema = tab.schema
        self.root.insert(tab)

    def insert_files(self, files, finalize=True):
        ingester = get_ingester(files)
        for tab in ingester:
            self.insert(tab)
        if finalize:
            self.finalize()

    def final_schema(self):
        """
        Returns the schema for all of the items in the quadtree, including written
        sidecars.
        """
        fields = [*self.read_root_table().schema]
        for sidecarname in set(self.sidecars.values()):
            sidecar = self.read_root_table(sidecarname).schema
            fields = [*fields, *sidecar]
        return pa.schema(fields)

    def read_root_table(self, suffix: Optional[str] = ""):
        path = self.basedir / "0/0/0"
        if suffix != "" and suffix is not None:
            path = path.with_suffix(f".{suffix}.feather")
        else:
            path = path.with_suffix(".feather")
        return feather.read_table(path)

    def finalize(self):
        manifest = self.root.finalize()
        flattened = flatten_manifest(manifest)
        tb = pa.Table.from_pylist(flattened).replace_schema_metadata(
            {
                "sidecars": json.dumps(self.sidecars),
                # The complete schema includes
                "schema": self.final_schema().serialize().to_pybytes(),
            }
        )
        feather.write_feather(
            tb, self.basedir / "manifest.feather", compression="uncompressed"
        )

    @property
    def manifest_table(self):
        return feather.read_table(self.basedir / "manifest.feather")

    def build_bloom_index(self, id_field: str, id_sidecar: Optional[str], m=28, k=10):
        """
        Creates an index of bloom filters.
        """
        # 24 and 11 are made up defaults here. I chose them
        # because I was on a plane and gpt4all suggested 20, 7.

        # Make id_field safe for filenames
        for macrotile in self.macrotiles:
            macrotile.build_bloom_filter(id_field, m, k)

    def join(
        self,
        data: Iterator[pa.Table],
        id_field: str,
        new_sidecar_name: str,
        m=24,
        k=2,
    ):
        """
        Given an iterator of data with a keyed field 'key' already present,
        creates matched sidecars.
        """
        root = self.root_macrotile
        if root is None:
            raise ValueError("No root macrotile found.")
        for tb in data:
            if isinstance(tb, pa.RecordBatch):
                tb = pa.Table.from_batches([tb])
            root.insert_for_join(tb, id_field, m, k, new_sidecar_name)
        root.complete_insert_stage(id_field, m, k, new_sidecar_name)
        root.complete_join(id_field, m, k, new_sidecar_name)

    @property
    def root_macrotile(self):
        # It's just the first one in the iterated list.
        for file in self.macrotiles:
            if file.coords == (0, 0, 0):
                return file
        raise ValueError("No root macrotile found.")

    def iterate(
        self,
        top_down: bool = True,
        breadth_firth: bool = True,
        mode: Literal["read", "append"] = "read",
    ) -> Iterator["Tile"]:
        """
        Iterates through all the tiles in a constructed tree, bottom-up
        or top-down.
        """
        files = self.manifest_table.to_pylist()
        lookups: Dict[Tuple[int, ...], Rectangle] = {}
        for f in files:
            k = tuple(map(int, f["key"].split("/")))
            assert len(k) == 3
            r = json.loads(f["extent"])
            lookups[k] = Rectangle(x=(r["x"][0], r["x"][1]), y=(r["y"][0], r["y"][1]))
        to_check: List[Tuple[int, int, int]] = [(0, 0, 0)]
        ordered_list = []
        while len(to_check) > 0:
            z, x, y = to_check.pop(0)
            if (z, x, y) in lookups:
                ordered_list.append((z, x, y))
                possible_children = children((z, x, y))
                if breadth_firth:
                    to_check.extend(possible_children)
                else:
                    to_check = [*possible_children, *to_check]
        if not top_down:
            ordered_list.reverse()
        for key in ordered_list:
            yield Tile(self, extent=lookups[key], tile_code=key, mode=mode)

    def iter_macrotile_order(
        self,
    ) -> Iterator[Tuple[Tuple[int, int, int], List["Tile"]]]:
        macros = defaultdict(list)
        for tile in self.iterate(top_down=True):
            macros[macrotile(tile.coords)].append(tile)
        for coords, tiles in macros.items():
            yield (coords, tiles)

    def iter_macrotiles(self):
        for key, tiles in self.iter_macrotile_order():
            t = Macrotile(self, key, tiles)
            yield t


class Macrotile:
    """
    This is an experimental class for operations that are batched across multiple tiles.

    Currently it's only used for bloom filters. A macrotile consists of 21 tiles; a root,
    its four children, and its sixteen grandchildren. each of the grandchildren is the
    root of its *own* macrotile.
    """

    def __init__(
        self,
        quadtree: Quadtree,
        coords: Tuple[int, int, int],
        tiles: Optional[List["Tile"]] = None,
    ):
        # gut check -- macrotiles can never be 1/0/0, 3/1/3, etc.
        assert coords[0] % 2 == 0
        self.quadtree = quadtree
        self.coords = coords
        self._tiles = tiles
        self._open_filehandles: Dict[
            Path, Union[ipc.RecordBatchFileWriter, pq.ParquetWriter]
        ] = {}

    def write_batch_to_filehandle(self, path: Path, batch: pa.Table):
        if not path in self._open_filehandles:
            if path.suffix == ".feather":
                self._open_filehandles[path] = ipc.new_file(path, schema=batch.schema)
            elif path.suffix == ".parquet":
                self._open_filehandles[path] = pq.ParquetWriter(
                    path, schema=batch.schema
                )
            else:
                raise NotImplementedError(
                    path.suffix + " is not a supported file format"
                )
        # will fail if the schema changes.
        self._open_filehandles[path].write_table(batch)

    def children(self) -> List["Macrotile"]:
        potential_children = []
        for child in children(self.coords):
            for grandchild in children(child):
                potential_children.append(grandchild)
        all_macrotiles = {mt.coords: mt for mt in self.quadtree.macrotiles}
        actual_children = []
        for coords in potential_children:
            if coords in all_macrotiles:
                actual_children.append(all_macrotiles[coords])
        return actual_children

    def tiles(self) -> Iterator["Tile"]:
        for tile in self.quadtree.iterate(top_down=True):
            if macrotile(tile.coords) == self.coords:
                yield tile

    def bloom_filter_loc(self, id_field: str, m: int, k: int):
        """
        The location for a bloom filter about field 'id_field' with log2(m) bits and k hashes.
        """
        (z, x, y) = self.coords
        # Make sure the id_field is filename safe
        id_field_enc = urlsafe_b64encode(id_field.encode("utf-8")).decode("utf-8")
        bloom_filter_loc = (
            self.quadtree.basedir
            / "bloom_filters"
            / f"{m}-{k}"
            / f"{z}/{x}/{y}.{id_field_enc}.feather"
        )
        return bloom_filter_loc

    def build_bloom_filter(self, id_field: str, m: int, k: int):
        bloom_filter_loc = self.bloom_filter_loc(id_field, m, k)
        bloom_filter_loc.parent.mkdir(parents=True, exist_ok=True)
        if bloom_filter_loc.exists():
            return

        # Use a bool8 in numpy to actually build the filter.
        # this takes 8x as much space as we need, but IDK a robust
        # fast way to twiddle individual bits in python.

        # At *read* time, we are carefuly to use bitmasks.

        positions = np.zeros((2**m), np.bool)
        tilenames = []
        id_sidecar = self.quadtree.sidecars.get(id_field, None)

        for tile in self.tiles():
            col = tile.read_column(id_field, id_sidecar)
            # Integers are hashed as strings. Not ideal.
            if not pa.types.is_string(col.type):
                col = col.cast(pa.string())

            # TODO: This could probably be vectorized to work more
            # more efficiently; rather than set the bits one row at a time,
            # build of the list of all positions and set them at once.

            for item in col:
                # Retrieve the bytes of the string.
                bytes = item.as_buffer().to_pybytes()
                # Hash it with md5
                hashed = bloom_hash(bytes, m, k)
                # Set the k positions in the fiter to true.
                positions[hashed] = True
            tilenames.append(tile.key)

        (z, x, y) = self.coords

        tb = pa.table(
            {
                "key": pa.array([f"{z}/{x}/{y}"]),
                "tiles": pa.array([[tilenames]]),
                "bitmask": pa.array([positions], pa.list_(pa.bool_(), 2**m)),
            }
        )

        # We write records of a single row at a time. This is wasteful,
        # but the reason we're using arrow at all for these is to get bitmask-
        # lists in a sane form, so it's not the end of the world.

        self.bloom_filter_loc(id_field, m, k).parent.mkdir(parents=True, exist_ok=True)
        feather.write_feather(tb, bloom_filter_loc, compression="zstd")

    def matched_file_loc(self, new_sidecar_name, m, k):
        """
        The location to which we write files that match the bloom filter here.
        """
        return self.bloom_filter_loc(new_sidecar_name, m, k).with_suffix(
            ".matches.parquet"
        )

    def candidate_file_loc(self, new_sidecar_name, m, k):
        """
        Differs from 'matched_file_loc' because this file applies not just the
        macrotiles filter, but filters for all macrotiles *beneath* it.
        """
        return self.bloom_filter_loc(new_sidecar_name, m, k).with_suffix(
            ".candidates.parquet"
        )

    def bloom_filter(self, id_field, m, k):
        loc = self.bloom_filter_loc(id_field, m, k)
        if not loc.exists():
            logger.debug(("MAKING BLOOM FILTER", self.coords, id_field, m, k))
            self.build_bloom_filter(id_field, m, k)
        tb = feather.read_table(loc)
        return pc.list_flatten(tb.take([0])["bitmask"])

    def bloom_filters_below_here(self, id_field, m, k, inclusive=True):
        """
        Returns a single bloom filter which combines together all bloom filters
        below, including this one.
        """
        if inclusive:
            total_filter = self.bloom_filter(id_field, m, k)
        else:
            raise NotImplementedError("Not implemented")
            total_filter = pa.array(np.zeros(2**m), pa.bool_())
        children = self.children()
        while len(children) > 0:
            child = children.pop()
            filter = child.bloom_filter(id_field, m, k)
            total_filter = pc.or_(total_filter, filter)
            # Descend the tree
            children = [*children, *child.children()]
        return total_filter

    def insert_for_join(
        self, tb: pa.Table, id_field, m: int, k: int, new_sidecar_name: str
    ):
        """

        The first step in a join. This can be done any number of times.

        key: the join key.

        """
        assert id_field in tb.column_names, "Table must include identifier key"

        # We're going to divide the data across 21 possible places.
        # 1. A candidate file for this macrotile, because.
        # 2. A candidate file for the macrotile of the children of this tile
        # 3. A candidate file for each of the grandchildren of this tile, INCLUSIVE
        #    of any of their children.
        # Since Bloom filters are not exact, it is possible and expected that
        # a row might be written to multiple output locations.

        # At m = 24, this will take 336 MB of memory to hold the filters.
        # at m = 28, this will take 5.3 GB of memory to hold the filters.

        tb = tb.filter(pc.invert(pc.is_null(tb[id_field])))
        id_col = tb[id_field]

        # Pre-allocated a table to hold the hash for each column.
        hashes = np.zeros((len(tb), k), np.uint32)
        if not pa.types.is_string(id_col.type):
            id_col = id_col.cast(pa.string())
        for i, item in enumerate(tb[id_field]):
            bytes = item.as_buffer().to_pybytes()
            hashes[i] = bloom_hash(bytes, m, k)

        hash_cols = pa.table({f"hash_{i}": pa.array(hashes[:, i]) for i in range(k)})
        # logger.debug(f"Inserting for join {(self.coords, id_field, len(tb), m, k)}")
        for macrotile in [self, *self.children()]:
            key = str(macrotile.bloom_filter_loc(id_field, m, k))
            if not key in self.quadtree._bloom_cache:
                self.quadtree._bloom_cache[key] = macrotile.bloom_filter(id_field, m, k)
            bloom_filter = self.quadtree._bloom_cache[key]
            matches = tb
            loc_hash_cols = hash_cols
            for i in range(k):
                # Go through the hashes in vectorized order.
                is_match = bloom_filter.take(loc_hash_cols[f"hash_{i}"])
                matches = matches.filter(is_match)
                loc_hash_cols = loc_hash_cols.filter(is_match)

            if len(matches) > 0:
                path = macrotile.matched_file_loc(new_sidecar_name, m, k)
                macrotile.write_batch_to_filehandle(path, matches)

        # for grandchildren, insert everything below here.
        for child in self.children():
            for grandchild in child.children():
                key = (
                    str(grandchild.bloom_filter_loc(id_field, m, k))
                    + "_with_descendants"
                )
                if not key in self.quadtree._bloom_cache:
                    self.quadtree._bloom_cache[key] = (
                        grandchild.bloom_filters_below_here(id_field, m, k)
                    )

                bloom_filter = self.quadtree._bloom_cache[key]
                matches = tb
                loc_hash_cols = hash_cols
                for i in range(k):
                    # Go through the hashes in vectorized order.
                    is_match = bloom_filter.take(loc_hash_cols[f"hash_{i}"])
                    matches = matches.filter(is_match)
                    loc_hash_cols = loc_hash_cols.filter(is_match)
                if len(matches) > 0:
                    path = grandchild.candidate_file_loc(new_sidecar_name, m, k)
                    macrotile.write_batch_to_filehandle(path, matches)

    def close_filehandles(self, recursive=True):
        """
        Closes any open files below this point in the tree.

        Called in complete_insert_stage.
        """
        for path, file in [*self._open_filehandles.items()]:
            file.close()
            del self._open_filehandles[path]
        if recursive:
            for child in self.children():
                child.close_filehandles()

    def complete_insert_stage(self, id_field, m: int, k: int, new_sidecar_name):
        """
        The second step in a join. Finalizes the insertion recursively below, and
        closes any remaining resources.
        """
        self.close_filehandles(recursive=True)
        # Close the lagging bloom filters
        self.quadtree._bloom_cache = {}

        if self.candidate_file_loc(new_sidecar_name, m, k).exists():
            # "Candidate file loc" means that we encountered a bloom filter match
            # including all of the children of this tile.
            if len(self.children()) == 0:
                # If we have preliminary unmerged files but no children,
                # this means that we're done inserting, and can rename it
                # to be a 'matched' file.
                self.candidate_file_loc(new_sidecar_name, m, k).rename(
                    self.matched_file_loc(new_sidecar_name, m, k)
                )
            else:
                # Otherwise, we need to recursively insert that file
                # at an appropriate point in the tree.
                for batch in pq.ParquetFile(
                    self.candidate_file_loc(new_sidecar_name, m, k)
                ).iter_batches():
                    b = pa.Table.from_batches([batch])
                    self.insert_for_join(
                        b,
                        id_field,
                        m,
                        k,
                        new_sidecar_name,
                    )
                # Clean up the candidate file, because it has now been inserted below here.
                self.candidate_file_loc(new_sidecar_name, m, k).unlink()
                # Close the filehandles for the children immediately to avoid
                # lagging writers
                self.close_filehandles()
                self.quadtree._bloom_cache = {}
        for child in self.children():
            # Finally, we need to recursively complete the insert stage for all children.
            # This ensure that there all candidate files have been flushed to the farthest leaves
            # of the quadtree, that all files are closed and that there are no lingering
            # resources.
            child.complete_insert_stage(id_field, m, k, new_sidecar_name)
        # I don't *think* this is necessary, but one last check to make sure that
        # all the filehandles are closed.
        self.close_filehandles()
        self.quadtree._bloom_cache = {}

    def complete_join(self, id_field, m, k, new_sidecar_name):
        """
        The third and final stage in a join.
        Called once all the record batches of a join have been inserted.
        Now that we have parquet files identifying for each macrotile of 21 tiles
        a set of candidate matches, we can now read those files and write out
        the final sidecar files in exactly the same order as the original files.
        """
        try:
            sidecar = self.quadtree.sidecars[id_field]
        except KeyError:
            sidecar = None

        if not self.matched_file_loc(new_sidecar_name, m, k).exists():
            # Can happen if there are no matches.
            pass

        else:
            for tile in self.tiles():
                dest = tile.filename.with_suffix(f".{new_sidecar_name}.feather")
                if dest.exists():
                    raise FileExistsError(f"File {dest} already exists.")
                    # logger.warning(f"File {dest} already exists.")
                    # continue
                ids = tile.read_column(id_field, sidecar)
                # Read *only* the relevant rows of the parquet file.
                # So long as the
                matches = pq.read_table(
                    self.matched_file_loc(new_sidecar_name, m, k),
                    filters=[(id_field, "in", ids)],
                )
                sort_indices = pc.index_in(ids, matches[id_field])

                # Now we reshuffle the matches to be in the same order as the original file.

                towrite = matches.drop([id_field]).take(sort_indices)
                try:
                    # Sidecars should generally be a single record batch, but if they
                    # contain more than 2GB of text this may break.
                    towrite = towrite.combine_chunks()
                except pa.lib.ArrowInvalid:
                    logger.warning(f"Failed to combine chunks for {tile.coords}")
                feather.write_feather(
                    towrite,
                    dest,
                    compression="uncompressed",
                )
        for mychild in self.children():
            mychild.complete_join(id_field, m, k, new_sidecar_name)


def bloom_hash(val: bytes, m: int, k: int) -> np.ndarray:
    """
    Hash bytes
    Hashes a value val in k buckets out of 2**m total buckets.

    returns: a numpy integer array of length k indicating which numbers are set.
    """

    # Rather than actually hash m times for the bloom filter, we make a single
    # md5 hash which is 32 bytes, and then take slices of that offset by one byte
    # at a time to represent the different hash functions. So if there are
    #
    # Should this be 4 or 8?
    each_takes = int(m / 4)
    assert (
        32 - each_takes
    ) > k, f"cannot have more than {(32 - each_takes)} hashes in bloom index"

    vals = np.zeros(k, np.uint32)
    hashed = hashlib.md5(val).hexdigest()
    # Since we're
    for i in range(k):
        hash = int(hashed[i : i + each_takes], 16)
        vals[i] = hash
    return vals


def children(coords: Tuple[int, int, int]):
    z, x, y = coords
    return [(z + 1, x * 2 + i, y * 2 + j) for i in (0, 1) for j in (0, 1)]


def parent(coords: Tuple[int, int, int]):
    z, x, y = coords
    return (z - 1, x // 2, y // 2)


def macrotile(coords: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if coords[0] == 0:
        return (0, 0, 0)
    if coords[0] % 2 == 0:
        return parent(parent(coords))
    return parent(coords)


class Tile:
    """
    A tile is a node in the quadtree that has an associated record batch.

    Insert and build operations are recursive.
    """

    def __init__(
        self,
        quadtree: Quadtree,
        extent: Union[Rectangle, Tuple[Tuple[float, float], Tuple[float, float]]],
        tile_code: Tuple[int, int, int] = (0, 0, 0),
        permitted_children: Optional[int] = 4,
        # Actually dict of pa type constructors.
        parent: Optional["Tile"] = None,
        mode: Literal["read", "write", "append"] = "read",
    ):
        """
        Create a tile.

        destination

        """
        self.coords = tile_code
        self.quadtree = quadtree
        if isinstance(extent, tuple):
            extent = Rectangle(x=extent[0], y=extent[1])
        self.min_ix = 1_000_000_000_000
        self.max_ix = -1
        self.extent = extent
        self.schema = None
        self.permitted_children = permitted_children
        # Wait to actually create the directories until needed.
        self._filename: Optional[Path] = None
        self._children: Union[List[Tile], None] = None
        self._overflow_writers = None
        self._sinks = None
        self.parent = parent
        # Running count of inserted points. Used to create counters.
        self.count_inserted = 0
        # Unwritten records for that need to be flushed.
        self.data: Optional[List[pa.RecordBatch]] = []

        self.n_data_points: int = 0
        self.n_flushed = 0
        self.total_points = 0
        self.manifest: Optional[TileManifest] = None

    @property
    def tile_size(self):
        if self.coords[0] == 0:
            return self.quadtree.first_tile_size
        if self.coords[0] == 1:
            return int(sqrt(self.quadtree.tile_size * self.quadtree.first_tile_size))
        return self.quadtree.tile_size

    @property
    def sidecar_names(self):
        return set(self.quadtree.sidecars.values())

    @property
    def dictionaries(self):
        return self.quadtree.dictionaries

    def overall_tile_budget(self):
        return self.quadtree.max_open_filehandles

    def insert(
        self,
        tab: Union[pa.Table, pa.RecordBatch],
    ):
        if isinstance(tab, pa.RecordBatch):
            tab = pa.Table.from_batches([tab])
        if not "ix" in tab.schema.names:
            tab = tab.append_column(
                "ix",
                pa.array(
                    range(self.count_inserted, self.count_inserted + len(tab)),
                    pa.uint64(),
                ),
            )
        self.count_inserted += len(tab)
        self.min_ix = min(self.min_ix, tab["ix"][0].as_py())
        self.max_ix = max(self.max_ix, tab["ix"][-1].as_py())
        # A remapped version of the tile.
        d = dict()
        # Remap values if necessary.
        for t in tab.schema.names:
            if t in self.dictionaries:
                d[t] = remap_dictionary(tab[t], self.dictionaries[t])
            elif t in self.quadtree.schema.names:
                d[t] = pc.cast(tab[t], self.quadtree.schema.field(t).type, safe=False)
            else:
                d[t] = tab[t]

        if self.quadtree.randomize > 0:
            # Optional circular jitter to avoid overplotting.
            rho = nprandom.normal(0, self.quadtree.randomize, tab.shape[0])
            theta = nprandom.uniform(0, 2 * np.pi, tab.shape[0])
            d["x"] = pc.add(d["x"], pc.multiply(rho, pc.cos(theta)))
            d["y"] = pc.add(d["y"], pc.multiply(rho, pc.sin(theta)))
        tab = pa.table(d, schema=self.schema)
        self.schema = tab.schema
        self.insert_table(tab)

    @property
    def key(self):
        return "/".join(map(str, self.coords))

    def finalize(self):
        """
        Completes the quadtree so that no more points can be added.
        """
        logger.debug(f"first flush of {self.coords} complete")
        for tile in self.iterate(direction="top-down"):
            # First, we close any existing overflow buffers.
            tile.close_overflow_buffers()
        for tile in self.iterate(direction="top-down"):
            if tile.overflow_loc.exists():
                # Now we can begin again with the overall budget inserting at this point.
                tile.permitted_children = self.overall_tile_budget()
                tables = [
                    tile.overflow_loc,
                    *[
                        tile.overflow_loc.with_suffix(f".{k}.arrow")
                        for k in self.sidecar_names
                    ],
                ]
                inputs = [pa.ipc.open_file(tb) for tb in tables]

                def yielder() -> Iterator[pa.RecordBatch]:
                    for i in range(inputs[0].num_record_batches):
                        root = pa.Table.from_batches([inputs[0].get_batch(i)])
                        for sidecar in inputs[1:]:
                            batch = sidecar.get_batch(i)
                            for col in batch.column_names:
                                root = root.append_column(col, batch[col])
                        yield root.combine_chunks().to_batches()[0]

                # Insert in chunks of 100 megabytes
                for batch in rebatch(yielder(), 100e6):
                    tile.insert(batch)
                tile.overflow_loc.unlink()
                # Manifest isn't used
                tile.finalize()

        logger.debug(f"child insertion from {self.coords} complete")
        n_complete = 0
        # At the end, we move from the bottom-up and determine the children
        # for each point
        for tile in self.iterate(direction="bottom-up"):
            manifest = tile.final_flush()
            n_complete += 1
        return manifest  # is now the tile manifest for the root.

    def __repr__(self):
        return f"Tile:\nextent: {self.extent}\ncoordinates:{self.coords}"

    def midpoints(self) -> List[Tuple[str, float]]:
        midpoints: List[Tuple[str, float]] = []
        for k, lim in [("x", self.extent.x), ("y", self.extent.y)]:
            midpoint = (lim[1] + lim[0]) / 2
            midpoints.append((k, midpoint))
        # Ensure x,y,z order--shouldn't be necessary.
        midpoints.sort()
        return midpoints

    @property
    def filename(self) -> Path:
        if self._filename is not None:
            return self._filename
        local_name = Path(*map(str, self.coords)).with_suffix(".feather")
        dest_file = self.quadtree.basedir / local_name
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        self._filename = dest_file
        return self._filename

    def first_flush(self):
        # Ensure this function is only ever called once.
        if self.data is not None and len(self.data) > 0:
            destination = self.filename
            self.flush_data(destination, compression="uncompressed")
            self.data = None

    def close_overflow_buffers(self):
        self.first_flush()
        # Both will be the case together, checking them for type checks.
        if self._overflow_writers is not None and self._sinks is not None:
            for k, writer in self._overflow_writers.items():
                writer.close()
                self._sinks[k].close()
            self._overflow_writers = None
            self._sinks = None

    def final_flush(self) -> TileManifest:
        """
        At the end, we can see which tiles have children and append that
        to the metadata.

        Returns the number of rows below this point.
        """

        # This is the way that we can tell if we've already flushed.
        if self.manifest is not None:
            return self.manifest
        # extent = {k: [float(f) for f in v] for k, v in self.extent.items()}
        extent = Rectangle(
            x=(float(self.extent.x[0]), float(self.extent.x[1])),
            y=(float(self.extent.y[0]), float(self.extent.y[1])),
        )
        children = []
        if self._children is not None:
            for child in self._children:
                if child.total_points > 0:
                    if child.manifest is None:
                        raise ValueError("Child has not been flushed.")
                    self.total_points += child.total_points
                    children.append(child.manifest)
        min_ix, max_ix = self.min_ix, self.max_ix

        self.total_points += self.n_data_points
        self.manifest = TileManifest(
            key=self.id,
            nPoints=self.n_data_points,
            children=children,
            min_ix=min_ix,
            max_ix=max_ix,
            extent=extent,
        )
        return self.manifest

    def flush_data(self, destination, compression):
        """
        Flushes the locally stored data on this tile to disk

        Returns the number of points written.
        """
        if self.data is None:
            return 0

        schema_copy = pa.schema(self.data[0].schema)
        frame = pa.Table.from_batches(self.data, schema_copy).combine_chunks()
        other_tbs = defaultdict(dict)
        minmax = pc.min_max(frame["ix"]).as_py()

        for k, v in self.quadtree.sidecars.items():
            other_tbs[v][k] = frame[k]
            frame = frame.drop(k)
        feather.write_feather(frame, destination, compression=compression)
        for k, v in other_tbs.items():
            feather.write_feather(
                pa.table(v), destination.with_suffix(f".{k}.feather"), compression
            )
        self.min_ix = minmax["min"]
        self.max_ix = minmax["max"]
        self.data = None

    def insert_files(
        self,
        files,
        schema: Optional[pa.Schema] = None,
        destructively=False,
        finalize: bool = True,
    ):
        """
        given a list of files, insert them into the tree at this point.
        """
        ingester = get_ingester(files, destructive=destructively)
        if schema is None:
            schema = self.schema
        elif type(schema) == dict:
            schema = pa.schema(schema)
        logger.debug(f"starting insertion at {self.coords}")
        self.insert_ingester(ingester, finalize=finalize)

    def insert_ingester(self, ingester, finalize: bool = False):
        for tab in ingester:
            self.insert(tab)
        logger.debug(f"starting flush from {self.coords}")
        if finalize:
            self.finalize()

    @property
    def overflow_loc(self):
        return self.filename.with_suffix(".overflow.arrow")

    @property
    def overflow_buffers(self):
        """
        Creates or returns an arbitrary-length file.
        """

        if self._overflow_writers:
            return self._overflow_writers
        logger.debug(f"Opening overflow on {self.coords}")
        if self.overflow_loc.exists():
            raise FileExistsError(f"Overflow file already exists: {self.overflow_loc}")
        self._overflow_writers = {}
        self._sinks = {}
        for k in set(["", *self.quadtree.sidecars.values()]):
            assert self.quadtree.schemas[k] is not None
            p = self.overflow_loc
            if k != "":
                p = p.with_suffix(f".{k}.arrow")
            self._sinks[k] = pa.OSFile(str(p), "wb")
            self._overflow_writers[k] = pa.ipc.new_file(
                self._sinks[k], self.quadtree.schemas[k]
            )
        return self._overflow_writers

    def partition_to_children(self, table) -> List[pa.Table]:
        # Coerce to a list in quadtree/octree order.
        frames = [table]
        pivot_dim: Tuple[str, float]
        for pivot_dim in self.midpoints():
            expanded = []
            for frame in frames:
                expanded += partition(frame, pivot_dim)
            frames = expanded
        return frames

    @property
    def children(self):
        if self._children is not None:
            return self._children
        raise ValueError("Children have not been created.")

    def make_children(self, weights: List[float]):
        logger.debug(
            f"{' ' * self.coords[0]} Making children with budget of {self.permitted_children} for {self.key}, weights of {weights}"
        )
        # QUAD ONLY
        # Weights: a set of weights for how many children to create the
        # next level with.

        # Calling this forces child creation even when it's not wise.
        self._children = []
        midpoints = self.midpoints()

        # This tile has a budget. Four of those are spent on its children;
        # its largest children may also be allowed to procreate.
        permitted_grandchildren = self.permitted_children - 4

        child_permission = np.array([0, 0, 0, 0], np.int16)

        while permitted_grandchildren >= 4:
            biggest_child = np.argmax(weights)
            child_permission[biggest_child] += 4
            permitted_grandchildren -= 4
            weights[biggest_child] /= 4

        for i in [0, 1]:
            xlim = [midpoints[0][1], self.extent.x[i]]
            xlim.sort()
            for j in [0, 1]:
                ylim = [midpoints[1][1], self.extent.y[j]]
                ylim.sort()
                extent = Rectangle(x=(xlim[0], xlim[1]), y=(ylim[0], ylim[1]))
                coords = (
                    self.coords[0] + 1,
                    self.coords[1] * 2 + i,
                    self.coords[2] * 2 + j,
                )
                child = Tile(
                    quadtree=self.quadtree,
                    extent=extent,
                    tile_code=coords,
                    parent=self,
                    permitted_children=child_permission[i * 2 + j],
                )
                self._children.append(child)
        return self._children

    def iterate(self, direction):
        # iterate over the children in the given direction.
        assert direction in ["top-down", "bottom-up"]
        if direction == "top-down":
            yield self
        if self._children is not None:
            for child in self._children:
                yield from child.iterate(direction)
        if direction == "bottom-up":
            yield self

    @property
    def id(self):
        return "/".join(map(str, self.coords))

    def check_schema(self, table):
        if self.schema is None:
            self.schema = table.schema
            return
        if not self.schema.equals(table.schema):
            logger.debug("INSERTING:", table.schema)
            logger.debug("OWN SCHEMA", self.schema)
            raise TypeError("Attempted to insert a table with a different schema.")

    def insert_table(self, table: pa.Table):
        self.check_schema(table)
        insert_n_locally = self.tile_size - self.n_data_points
        if insert_n_locally > 0:
            if self.data is None:
                raise ValueError("Data was already flushed, failing")
            local_mask = np.zeros((table.shape[0]), bool)
            local_mask[:insert_n_locally] = True
            head = pc.filter(table, local_mask)
            if head.shape[0]:
                for batch in head.to_batches():
                    self.data.append(batch)
                    self.n_data_points += len(batch)
            # All *unwritten* rows are now mapped into the main table to
            # flow into the children.
            table = table.filter(pc.invert(local_mask))
            if self.n_data_points == self.tile_size:
                # We've just completed. Flush.
                self.flush_data(self.filename, "uncompressed")
        else:
            pass

        # 4, for quadtrees.
        children_per_tile = 2 ** (len(self.coords) - 1)

        # Always use previously created _overflow writer
        if not self._overflow_writers and (
            self.permitted_children >= children_per_tile or self._children is not None
        ):
            # If we can afford to create children, do so.
            total_records = table.shape[0]
            if total_records == 0:
                return
            partitioning = self.partition_to_children(table)
            # The next block creates children and uses up some of the budget:
            # This one accounts for it.
            if self._children is None:
                weights = [float(len(subset)) for subset in partitioning]
                self.make_children(weights)
            for child_tile, subset in zip(self.children, partitioning):
                child_tile.insert_table(subset)
        else:
            for sidecar, tb in self.keyed_batches(table).items():
                for tb in rebatch(tb.to_batches(), 50e6):
                    for batch in tb.to_batches():
                        self.overflow_buffers[sidecar].write_batch(batch)

    def keyed_batches(self, table: pa.Table):
        """
        Divides a table in the separate-sidecar tables to which it
        will be written.
        """
        cars = ["", *self.quadtree.sidecars.values()]
        schemas = self.quadtree.schemas
        return {k: table.select(schemas[k].names) for k in cars}

    def read_column(self, column: str, sidecar: Optional[str]) -> pa.Array:
        fname = self.filename
        if sidecar is not None:
            fname = self.filename.with_suffix(f".{sidecar}.feather")
        return feather.read_table(fname, columns=[column])[column]


def get_better_codes(col, counter=Counter()):
    for a in pc.value_counts(col):
        counter[a["values"].as_py()] += a["counts"].as_py()
    return counter


def get_recoding_arrays(files, col_name):
    countered = Counter()
    ingester = get_ingester(files, columns=[col_name])
    for batch in ingester:
        col = batch[col_name]
        countered = get_better_codes(col, countered)
    new_order = [pair[0] if pair[0] else "<NA>" for pair in countered.most_common(4094)]
    if len(new_order) == 4094:
        new_order.append("<Other>")
    new_order_dict = dict(zip(new_order, range(len(new_order))))
    return new_order, new_order_dict


def remap_dictionary(chunk, new_order):
    # Switch a dictionary to use a pre-assigned set of keys. returns a new chunked dictionary array.

    new_indices = pc.index_in(chunk, new_order)
    return pa.DictionaryArray.from_arrays(new_indices, new_order)


def cli():
    args = parse_args()
    if args["log_level"]:
        logger.setLevel(args["log_level"])
        del args["log_level"]

    main(**args)


if __name__ == "__main__":
    cli()
