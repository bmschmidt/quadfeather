import pyarrow as pa
from pyarrow import csv, feather, parquet as pq, compute as pc, ipc as ipc

# import pandas as pd
import logging
import shutil
from pathlib import Path
import sys
import argparse
import json
from numpy import random as nprandom
from collections import defaultdict, Counter
from typing import (
    DefaultDict,
    Iterator,
    Dict,
    List,
    Tuple,
    Set,
    Optional,
    Any,
    Union,
    Tuple,
)
import numpy as np
import sys
from math import isfinite, sqrt
from .ingester import get_ingester, Ingester
from dataclasses import dataclass, field

logger = logging.getLogger("quadfeather")

DEFAULTS: Dict[str, Any] = {
    "first_tile_size": 1000,
    "tile_size": 50000,
    "destination": Path("."),
    "max_files": 25.0,
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
        "--max_files",
        type=float,
        default=DEFAULTS["max_files"],
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
    logger.setLevel(args.log_level)

    return {**DEFAULTS, **vars(args)}


# First, we guess at the schema using pyarrow's own type hints.
# This could be overridden by user args.
def refine_schema(schema: pa.Schema) -> Dict[str, pa.DataType]:
    """ " """
    fields = {}
    seen_ix = False
    for el in schema:
        if isinstance(el.type, pa.DictionaryType) and pa.types.is_string(
            el.type.value_type
        ):
            t = pa.dictionary(pa.int16(), pa.utf8())
            fields[el.name] = t
        #            el = pa.field(el.name, t)
        elif pa.types.is_float64(el.type) or pa.types.is_float32(el.type):
            fields[el.name] = pa.float32()
        elif el.name == "ix":
            fields[el.name] = pa.uint64()
            seen_ix = True
        elif pa.types.is_integer(el.type):
            # Integers become float32
            fields[el.name] = pa.float32()
        elif pa.types.is_large_string(el.type) or pa.types.is_string(el.type):
            fields[el.name] = pa.string()
        elif pa.types.is_boolean(el.type):
            fields[el.name] = pa.float32()
        elif pa.types.is_date32(el.type):
            fields[el.name] = pa.date32()
        elif pa.types.is_temporal(el.type):
            fields[el.name] = pa.timestamp("ms")
        else:
            raise TypeError(f"Unsupported type {el.type}")
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
        if t == pa.int64() and el.name != "ix":
            t = pa.float32()
        if t == pa.int32() and el.name != "ix":
            t = pa.float32()
        if t == pa.float64():
            t = pa.float32()
        if t == pa.large_string():
            t = pa.string()
        if isinstance(t, pa.DictionaryType) and pa.types.is_string(t.value_type):
            t = pa.dictionary(pa.int16(), pa.utf8())
        schema[el.name] = t
        if el.name in override:
            schema[el.name] = getattr(pa, override[el.name])()
    schema["ix"] = pa.int64()
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
            logging.info(f"Batch no {chunk_num}")
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
    # Actually dict of pa type constructors.
    dtypes: Dict[str, Any] = {},
):
    """
    Run a tiler

    arguments: a list of strings to parse. If None, treat as command line args.
    csv_block_size: the block size to use when reading the csv files. The default should be fine,
        but included here to allow for testing multiple blocks.
    """

    files = [Path(file) for file in files]
    destination = Path(destination)
    if extent is not None and isinstance(extent, dict):
        extent = Rectangle(**extent)
    dirs_to_cleanup = []
    if files[0].suffix == ".csv" or str(files[0]).endswith(".csv.gz"):
        schema, schema_safe = determine_schema(files)
        # currently the dictionary type isn't supported while reading CSVs.
        # So we have to do some monkey business to store it as keys at first, then convert to dictionary later.
        logger.info(schema)
        rewritten_files, extent, raw_schema = rewrite_in_arrow_format(
            files, schema_safe, schema, csv_block_size
        )
        dirs_to_cleanup.append(rewritten_files[0].parent)
        logger.info("Done with preliminary build")
    else:
        rewritten_files = files
        if extent is None:
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

    logging.info("Starting.")

    recoders: Dict = dict()

    for field in raw_schema:
        if pa.types.is_dictionary(field.type):
            recoders[field.name] = get_recoding_arrays(rewritten_files, field.name)

    if extent is None:
        raise ValueError("Extent must be defined.")
    tiler = Tile(
        extent=extent,
        basedir=destination,
        tile_code=(0, 0, 0),
        first_tile_size=first_tile_size,
        tile_size=tile_size,
        dictionaries=dictionaries,
        dtypes=dtypes,
    )
    tiler.insert_files(files=rewritten_files, schema=schema, recoders=recoders)

    logger.info("Job complete.")


def partition(table: pa.Table, midpoint: Tuple[str, float]) -> List[pa.Table]:
    # Divide a table in two based on a midpoint
    key, pivot = midpoint
    criterion = pc.less(table[key], np.float32(pivot))
    splitted = [pc.filter(table, criterion), pc.filter(table, pc.invert(criterion))]
    return splitted


class Tile:
    """
    A tile is a node in the quadtree that has an associated record batch.

    Insert and build operations are recursive.
    """

    def __init__(
        self,
        extent: Union[Rectangle, Tuple[Tuple[float, float], Tuple[float, float]]],
        basedir: Path,
        tile_code: Tuple[int, int, int] = (0, 0, 0),
        first_tile_size=2000,
        tile_size=65_000,
        permitted_children: Optional[int] = 5,
        randomize=0.0,
        dictionaries: Dict[str, pa.Array] = {},
        # Actually dict of pa type constructors.
        dtypes: Dict[str, Any] = {},
        parent: Optional["Tile"] = None,
    ):
        """
        Create a tile.

        destination

        """
        self.ix_extent: Tuple[int, int] = (-1, -2)  # Initalize with bad values
        self.coords = tile_code

        self.dictionaries = dictionaries
        self.dtypes = dtypes
        if isinstance(extent, tuple):
            extent = Rectangle(x=extent[0], y=extent[1])
        self.extent = extent
        self.randomize = randomize
        self.basedir = basedir
        self.first_tile_size = first_tile_size
        self.tile_size = tile_size
        self.schema = None
        self.permitted_children = permitted_children
        # Wait to actually create the directories until needed.
        self._filename: Optional[Path] = None
        self._children: Union[List[Tile], None] = None
        self._overflow_writer = None
        self._sink = None
        self.parent = parent
        # Running count of inserted points. Used to create counters.
        self.count_inserted = 0
        # Unwritten records for that need to be flushed.
        self.data: Optional[List[pa.RecordBatch]] = []

        self.n_data_points: int = 0
        self.n_flushed = 0
        self.total_points = 0
        self.manifest: Optional[TileManifest] = None

    def overall_tile_budget(self):
        if self.parent is not None:
            return self.parent.overall_tile_budget()
        return self.permitted_children

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
        d = dict()
        # Remap values if necessary.
        for t in tab.schema.names:
            if t in self.dictionaries:
                d[t] = remap_dictionary(tab[t], self.dictionaries[t])
            elif t in self.dtypes:
                d[t] = pc.cast(tab[t], self.dtypes[t](), safe=False)
            else:
                d[t] = tab[t]

        if self.randomize > 0:
            # Optional circular jitter to avoid overplotting.
            rho = nprandom.normal(0, self.randomize, tab.shape[0])
            theta = nprandom.uniform(0, 2 * np.pi, tab.shape[0])
            d["x"] = pc.add(d["x"], pc.multiply(rho, pc.cos(theta)))
            d["y"] = pc.add(d["y"], pc.multiply(rho, pc.sin(theta)))
        tab = pa.table(d, schema=self.schema)
        self.schema = tab.schema
        self.insert_table(tab)

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
                input = pa.ipc.open_file(tile.overflow_loc)
                yielder = (input.get_batch(i) for i in range(input.num_record_batches))
                # Insert in chunks of 100M
                for batch in rebatch(yielder, 100e6):
                    tile.insert(batch)
                tile.overflow_loc.unlink()
                # Manifest isn't used
                manifest = tile.finalize()

        logger.debug(f"child insertion from {self.coords} complete")
        n_complete = 0
        # At the end, we move from the bottom-up and determine the children
        # for each point
        for tile in self.iterate(direction="bottom-up"):
            manifest = tile.final_flush()
            n_complete += 1
        if manifest.key == "0/0/0":
            flattened = flatten_manifest(manifest)
            tb = pa.Table.from_pylist(flattened)
            feather.write_feather(
                tb, self.basedir / "manifest.feather", compression="uncompressed"
            )
        return manifest  # is now the tile manifest for the root.

    def __repr__(self):
        return f"Tile:\nextent: {self.extent}\ncoordinates:{self.coords}"

    @property
    def TILE_SIZE(self):
        depth = self.coords[0]
        if depth == 0:
            return self.first_tile_size
        elif depth == 1:
            # geometric mean of first and second
            return int(sqrt(self.tile_size * self.first_tile_size))
        else:
            return self.tile_size

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
        dest_file = self.basedir / local_name
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
        if self._overflow_writer is not None and self._sink is not None:
            self._overflow_writer.close()
            self._sink.close()
            self._overflow_writer = None

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
        min_ix, max_ix = self.ix_extent

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
        Flushes the locally stored data to disk

        Returns the number of points written.
        """
        if self.data is None:
            return 0

        schema_copy = pa.schema(self.data[0].schema)
        frame = pa.Table.from_batches(self.data, schema_copy).combine_chunks()
        feather.write_feather(frame, destination, compression=compression)
        minmax = pc.min_max(frame["ix"]).as_py()
        self.min_ix = minmax["min"]
        self.max_ix = minmax["max"]
        self.data = None

    def insert_files(
        self,
        files,
        recoders={},
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
        self.insert_ingester(ingester, schema, recoders, finalize=finalize)

    def insert_ingester(
        self, ingester, schema=None, recoders={}, finalize: bool = False
    ):

        for tab in ingester:
            self.insert(tab)
        logger.debug(f"starting flush from {self.coords}")
        if finalize:
            self.finalize()

    @property
    def overflow_loc(self):
        return self.filename.with_suffix(".overflow.arrow")

    @property
    def overflow_buffer(self):
        """
        Creates or returns an arbitrary-length file.
        """

        if self._overflow_writer:
            return self._overflow_writer
        logger.debug(f"Opening overflow on {self.coords}")
        if self.overflow_loc.exists():
            raise FileExistsError(f"Overflow file already exists: {self.overflow_loc}")
        self._sink = pa.OSFile(str(self.overflow_loc), "wb")
        self._overflow_writer = pa.ipc.new_file(self._sink, self.schema)
        return self._overflow_writer

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
                tilesize = self.TILE_SIZE
                if coords[0] == 1:
                    tilesize = int(sqrt(self.first_tile_size * self.tile_size))
                child = Tile(
                    extent=extent,
                    basedir=self.basedir,
                    tile_code=coords,
                    parent=self,
                    tile_size=tilesize,
                    permitted_children=child_permission[i * 2 + j],
                    dtypes=self.dtypes,
                    dictionaries=self.dictionaries,
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
            logger.error("INSERTING:", table.schema)
            logger.error("OWN SCHEMA", self.schema)
            raise TypeError("Attempted to insert a table with a different schema.")

    def insert_table(self, table: pa.Table):
        self.check_schema(table)
        insert_n_locally = self.TILE_SIZE - self.n_data_points
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
            if self.n_data_points == self.TILE_SIZE:
                # We've just completed. Flush.
                self.flush_data(self.filename, "uncompressed")
        else:
            pass

        # 4, for quadtrees.
        children_per_tile = 2 ** (len(self.coords) - 1)

        # Always use previously created _overflow writer
        if not self._overflow_writer and (
            self.permitted_children >= children_per_tile or self._children is not None
        ):
            # If we can afford to create children, do so.
            total_records = table.shape[0]
            if total_records == 0:
                return
            partitioning = self.partition_to_children(table)
            # The next block creates children and uses up some of the budget:
            # This one accounts for it.

            weights = [float(len(subset)) for subset in partitioning]

            if self._children is None:
                self.make_children(weights)

            # amount_for_children = self.tile_budget

            for child_tile, subset in zip(self.children, partitioning):
                child_tile.insert_table(subset)
        else:
            for batch in table.to_batches():
                self.overflow_buffer.write_batch(batch)


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


def rebatch(input: Iterator[pa.RecordBatch], size: float) -> Iterator[pa.Table]:
    """
    Assembles an iterator over record batches into a predictable memory size

    Yields *more* than the requested size to avoid bad situations of zero-size arrays.
    """
    buffer = []
    buffer_size = 0
    for batch in input:
        buffer_size += batch.nbytes
        buffer.append(batch)
        if buffer_size > size:
            yield pa.Table.from_batches(buffer)
            buffer = []
            buffer_size = 0
    if len(buffer) > 0:
        yield pa.Table.from_batches(buffer)


if __name__ == "__main__":
    args = parse_args()
    main(**args)

    # def from_arrow_generator(
    #     self,
    #     ingester: Ingester,
    #     recoders={},
    #     schema=None,
    #     destructively=False,
    # ):
    #     pass

    # def insert_files(
    #     self,
    #     files,
    #     recoders={},
    #     schema: Optional[pa.Schema] = None,
    #     destructively=False,
    #     finalize: bool = False,
    # ):
    #     """
    #     given a list of files, insert them into the tree at this point.
    #     """
    #     ingester = get_ingester(files, destructive=destructively)
    #     if schema is None:
    #         schema = self.schema
    #     elif type(schema) == dict:
    #         schema = pa.schema(schema)
    #     logger.debug(f"starting insertion at {self.coords}")
    #     self.insert_ingester(ingester, schema, recoders, finalize=finalize)

    # def insert_ingester(
    #     self, ingester, schema=None, recoders={}, finalize: bool = False
    # ):

    #     for tab in ingester:
    #         self.insert(tab, recoders, finalize=finalize)
    #     logger.debug(f"starting flush from {self.coords}")
    #     if finalize:
    #         self.finalize()
