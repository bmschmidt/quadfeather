import pyarrow as pa
from pyarrow import csv, feather, parquet as pq, compute as pc, ipc as ipc
#import pandas as pd
import logging
import shutil
from pathlib import Path
import sys
import argparse
import json
from numpy import random as nprandom
from collections import defaultdict, Counter
from typing import DefaultDict, Dict, List, Tuple, Set, Optional
import numpy as np
import sys
from math import isfinite, sqrt
from .ingester import get_ingester
logger = logging.getLogger("quadfeather")

def parse_args(arguments = None):
    parser = argparse.ArgumentParser(description='Tile an input file into a large number of arrow files.')
    parser.add_argument('--first_tile_size', type=int, default = 1000, help ="Number of records in first tile.")

    parser.add_argument('--tile_size', type=int, default = 50000, help ="Number of records per tile.")
    parser.add_argument('--destination', '--directory', '-d', type=Path, required = True, help = "Destination directory to write to.")
    parser.add_argument('--max_files',
        type=float,
        default = 25,
        help = "Max files to have open at once. Default 25. "
               "Usually it's much faster to have significantly fewer files than the max allowed open by the system "
    )
    parser.add_argument('--randomize', type=float, default = 0,
                        help ="Uniform random noise to add to points. If you have millions "
                        "of coincident points, can reduce the depth of the tree greatly.")

    parser.add_argument('--files', "-f", nargs = "+",
                        type = Path,
                        help="""Input file(s). If column 'ix' defined, will be used: otherwise, indexes will be assigned. Must be in sorted order.""")

    parser.add_argument('--limits', nargs = 4,
                        type = float,
                        metavar = list,
                        default = [float("inf"), float("inf"), -float("inf"), -float("inf")],
                        help="""Data limits, in [x0 y0 xmax ymax] order. If not entered, will be calculated at some cost.""")

    parser.add_argument('--dtypes', nargs = "*",
                        type = str,
                        metavar = list,
                        default = [],
                        help="Datatypes to override, in 'key=value' format with no spaces. Eg --dtypes year=float32")

    parser.add_argument('--log-level',
        type = int,
        default = 30)
    if arguments is None:
        #normal argparse behavior
        arguments = sys.argv[1:]
    args = parser.parse_args(arguments)
    logger.setLevel(args.log_level)

    return args

# First, we guess at the schema using pyarrow's own type hints.

# This could be overridden by user args.

def refine_schema(schema : pa.Schema) -> Dict[str,pa.DataType]:
    """"

    """
    fields = {}
    seen_ix = False
    for el in schema:
        if isinstance(el.type, pa.DictionaryType) and pa.types.is_string(el.type.value_type):
            t = pa.dictionary(pa.int16(), pa.utf8())
            fields[el.name] = t
#            el = pa.field(el.name, t)
        elif pa.types.is_float64(el.type) or pa.types.is_float32(el.type):
            fields[el.name] = pa.float32()
        elif el.name == "ix":
            fields[el.name] = pa.uint64()
            seen_ix = True
        elif pa.types.is_integer(el.type):
            fields[el.name] = pa.float32()
        elif pa.types.is_large_string(el.type) or pa.types.is_string(el.type):
            fields[el.name] = pa.string()
        elif pa.types.is_boolean(el.type):
            fields[el.name] = pa.float32()
        elif pa.types.is_date32(el.type):
            fields[el.name] = pa.date32()
        elif pa.types.is_temporal(el.type):
            fields[el.name] = pa.timestamp('ms')
        else:
            raise TypeError(f"Unsupported type {el.type}")
            fields[el.name] = el.type
    if not seen_ix:
        fields["ix"] = pa.uint64()
    return fields

def determine_schema(args):
    vals = csv.open_csv(args.files[0],
                        csv.ReadOptions(block_size= 1024*1024*64),
                        convert_options = csv.ConvertOptions(
                            auto_dict_encode = True,
                            auto_dict_max_cardinality=4094
                        ))
    override = {}
    for arg in args.dtypes:
        k, v = arg.split("=")
        override[k] = v

    raw_schema = vals.read_next_batch().schema

    schema = {}
    for el in raw_schema:
        t = el.type
        if t == pa.int64() and el.name != 'ix':
            t = pa.float32()
        if t == pa.int32() and el.name != 'ix':
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
    schema['ix'] = pa.int64()
    schema_safe = dict([(k, v if not pa.types.is_dictionary(v) else pa.string()) for k, v in schema.items()])
    return schema, schema_safe


# Next, convert these CSVs into some preliminary arrow files.
# These are the things that we'll actually read in.

# We write to arrow because we need a first pass anyway to determine
# the data bounds and some other stuff; and it will be much faster to re-parse
# everything from arrow than from CSV.

def rewrite_in_arrow_format(files, schema_safe : pa.Schema, 
        schema : pa.Schema, csv_block_size : int = 1024*1024*128):
    # Returns: an extent and a list of feather files.
    # files: a list of csvs
    # schema_safe: the output schema (with no dictionary types)
    # schema: the input schema (with dictionary types)
    # csv_block_size: the block size to use when reading the csv files.

    ix = 0
    extent = {
        "x": [float("inf"), -float("inf")],
        "y": [float("inf"), -float("inf")],
    }
    output_dir = files[0].parent / "_deepscatter_tmp"
    output_dir.mkdir()
    if "z" in schema.keys():
        extent["z"] = [float("inf"), -float("inf")]

    rewritten_files : list[Path] = []
    for FIN in files:
        vals = csv.open_csv(FIN, csv.ReadOptions(block_size = csv_block_size),
                                convert_options = csv.ConvertOptions(
                                    column_types = schema_safe))
        for chunk_num, batch in enumerate(vals):
            logging.info(f"Batch no {chunk_num}")
            # Loop through the whole CSV, writing out 100 MB at a time,
            # and converting each batch to dictionary as we go.
            d = dict()
            for i, name in enumerate(batch.schema.names):
                if pa.types.is_dictionary(schema[name]): ###
                    d[name] = batch[i].dictionary_encode() ###
                else: ###
                    d[name] = batch[i] ###
            data = pa.Table.from_batches([batch])
            # Count each element in a uint64 array (float32 risks overflow,
            # Uint32 supports up to 2 billion or so, which is cutting it close for stars.)
            d["ix"] = pa.array(range(ix, ix + len(batch)), type = pa.uint64())
            ix += len(batch)
            for dim in extent.keys(): # ["x", "y", maybe "z"]
                col = data.column(dim)
                zoo = col.to_pandas().agg([min, max])
                extent[dim][0] = min(extent[dim][0], zoo['min'])
                extent[dim][1] = max(extent[dim][1], zoo['max'])
            final_table = pa.table(d)
            fout = output_dir / f"{chunk_num}.feather"
            feather.write_feather(final_table, fout, compression = "zstd")
            rewritten_files.append(fout)
    raw_schema = final_table.schema
    return rewritten_files, extent, raw_schema
    # Learn the schema from the last file written out.

def check_filesnames(args):
    files = args.files
    ftypes = [f.split(".")[-1] for f in files]
    for f in ftypes:
        if f != f[0]:
            raise TypeError(f"Must pass all the same type of file as input, not {f}/{f[0]}")
    if not f in set(["arrow", "feather", "csv", "gz"]):
        raise TypeError("Must use files ending in 'feather', 'arrow', or '.csv'")

# Start at one to count the initial tile. (This number is incremented when children are built.)
memory_tiles_open : Set[str] = set()
files_open : Set[str] = set()

# Will be overwritten from args


def main(arguments = None, csv_block_size = 1024*1024*128):
    """
    Run a tiler

    arguments: a list of strings to parse. If None, treat as command line args.
    csv_block_size: the block size to use when reading the csv files. The default should be fine,
        but included here to allow for testing multiple blocks.
    """
    args = parse_args(arguments)
    dirs_to_cleanup = []
    if (args.files[0].suffix == '.csv' or str(args.files[0]).endswith(".csv.gz")):
        schema, schema_safe = determine_schema(args)
        # currently the dictionary type isn't supported while reading CSVs.
        # So we have to do some monkey business to store it as keys at first, then convert to dictionary later.
        logger.info(schema)
        rewritten_files, extent, raw_schema = rewrite_in_arrow_format(args.files, schema_safe, schema, csv_block_size)
        dirs_to_cleanup.append(rewritten_files[0].parent)
        logger.info("Done with preliminary build")
    else:
        rewritten_files = args.files
        if not isfinite(args.limits[0]):
            fin = args.files[0]
            if fin.suffix == ".feather" or fin.suffix == ".parquet":
                reader = get_ingester(args.files)
                xmin = args.limits[0]
                ymin = args.limits[1]
                xmax = args.limits[2]
                ymax = args.limits[3]
                for batch in reader:
                    x = pc.min_max(batch['x']).as_py()
                    y = pc.min_max(batch['y']).as_py()
                    if x['min'] < xmin and isfinite(x['min']):
                        xmin = x['min']
                    if x['max'] > xmax and isfinite(x['max']):
                        xmax = x['max']
                    if y['min'] < ymin and isfinite(y['min']):
                        ymin = y['min']
                    if y['max'] > ymax and isfinite(y['max']):
                        ymax = y['max']
                extent = {
                    "x": [xmin, xmax],
                    "y": [ymin, ymax]
                }
        else:
            extent = {
                "x": [args.limits[0], args.limits[2]],
                "y": [args.limits[1], args.limits[3]]
            }

        logger.info("extent")
        logger.info(extent)
        if args.files[0].suffix == '.feather':
            first_batch = ipc.RecordBatchFileReader(args.files[0])
            raw_schema = first_batch.schema
        elif args.files[0].suffix == '.parquet':
            first_batch = pq.ParquetFile(args.files[0])
            raw_schema = first_batch.schema_arrow
        schema = refine_schema(raw_schema)
        elements = {name : type for name, type in schema.items()}
        if not "ix" in elements:
            # Must always be an ix field.
            elements['ix'] = pa.uint64()
        raw_schema = pa.schema(elements)


    logging.info("Starting.")

    count_holder = defaultdict(Counter)

    recoders : Dict = dict()

    for field in raw_schema:
        if (pa.types.is_dictionary(field.type)):
            recoders[field.name] = get_recoding_arrays(rewritten_files, field.name)

    count_read = 0

    tiler = Tile(extent, [0, 0, 0], args)
    tiler.insert_files(files = rewritten_files, schema = schema, recoders = recoders)

    logger.info("Job complete.")
    

def partition(table: pa.Table, midpoint: Tuple[str, float]) -> List[pa.Table]:
    # Divide a table in two based on a midpoint
    key, pivot = midpoint
    criterion = pc.less(table[key], np.float32(pivot))
    splitted = [pc.filter(table, criterion), pc.filter(table, pc.invert(criterion))]
    return splitted


class Tile():
    # some prep to make OCT_TREE SAFE--METHODS that support only quads
    # listed as QUAD_ONLY
    def __init__(self, extent, coords, args, writer_budget = None):

        self.coords = coords
        self.extent = extent
        self.args = args
        self.basedir = args.destination
        self.first_tile_size = args.first_tile_size
        self.tile_size = args.tile_size
        self.schema = None
        if writer_budget is None:
          self.writer_budget = args.max_files
        # Wait to actually create the directories until needed.
        self._filename = None
        self._children = None
        self._overflow_writer = None
        self._sink = None
        # Unwritten records for myself.
        self.data : List[pa.RecordBatch] = []
        # Unwritten records for my children.
        self.hold_for_children = []

        self.n_data_points = 0
        self.n_flushed = 0
        self.total_points = 0
    
    def insert_files(self, files, recoders = {}, schema = None, randomize = 0, destructively = False):
      """
      given a list of files, inser them into the tree at this point.
      """
      ingester = get_ingester(files, destructive = destructively)
      if schema is None:
        schema = self.schema
      elif type(schema == dict):
        schema = pa.schema(schema)
      logger.debug(f"starting insertion at {self.coords}")
      count_read = 0
      for i, tab in enumerate(ingester):
        # Would only happen on the first insertion, so safe to do in order.
        if not "ix" in tab.schema.names:
          tab = tab.append_column("ix", pa.array(range(count_read, count_read + len(tab)), pa.uint64()))
        count_read += len(tab)
        d = dict()
        # Remap values if necessary.
        for t in tab.schema.names:
            if t in recoders:
                d[t] = remap_all_dicts(tab[t], *recoders[t])
            d[t] = tab[t]
        if randomize > 0:
            # Circular jitter to avoid overplotting.
            rho = nprandom.normal(0, args.randomize, tab.shape[0])
            theta = nprandom.uniform(0, 2 * np.pi, tab.shape[0])
            d['x'] = pc.add(d['x'], pc.multiply(rho, pc.cos(theta)))
            d['y'] = pc.add(d['y'], pc.multiply(rho, pc.sin(theta)))
        tab = pa.table(d, schema = schema)
        self.insert_table(tab, tile_budget = self.args.max_files)
      logger.debug(f"starting flush from {self.coords}")

      for tile in self.iterate(direction = "top-down"):
        # Freeze local and overflow tables to disk.
        tile.first_flush()
      logger.debug(f"first flush of {self.coords} complete")
      for tile in self.iterate(direction = "top-down"):
        if tile.overflow_loc.exists():
          tile.insert_files([tile.overflow_loc], destructively = True)
      logger.debug(f"child insertion from {self.coords} complete")
      n_complete = 0
      for tile in self.iterate(direction = "bottom-up"):
        n_complete += tile.final_flush()
      logger.info(f"{n_complete} tiles flushed from {self.coords}")

    def __repr__(self):
        return f"Tile:\nextent: {self.extent}\ncoordinates:{self.coords}"

    @property
    def TILE_SIZE(self):
        depth = self.coords[0]
        if depth == 0:
          return self.args.first_tile_size
        elif depth == 1:
          # geometric mean of first and second
          return int(sqrt(self.args.tile_size * self.args.first_tile_size))
        else:
          return self.args.tile_size

    def midpoints(self) -> List[Tuple[str, float]]:
        midpoints : List[Tuple[str, float]] = []
        for k, lim in self.extent.items():
            # params = []
            midpoint = (lim[1] + lim[0])/2
            midpoints.append((k, midpoint))
        # Ensure x,y,z order--shouldn't be necessary.
        midpoints.sort()
        return midpoints

    @property
    def filename(self):
        if self._filename:
            return self._filename
        local_name = Path(*map(str, self.coords)).with_suffix(".feather")
        dest_file = Path(self.basedir) / local_name
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        self._filename = dest_file
        return self._filename

    def first_flush(self):
        # Ensure this function is only ever called once.
        if self.data is not None and len(self.data) > 0:
            destination = self.filename.with_suffix(".needs_metadata.feather")
            n_written = self.flush_data(destination, {}, "zstd")
            self.data = None
        if self._overflow_writer is not None:          
          self._overflow_writer.close()
          self._sink.close()
          self._overflow_writer = None

    def final_flush(self) -> int:
        """
        At the end, we can see which tiles have children and append that
        to the metadata.

        Returns the number of rows below this point.
        """
        if self.total_points > 0:
            return self.total_points
        metadata = {
            "extent": json.dumps({k : [float(f) for f in v] for k, v in self.extent.items()}),
        }

        if self._children is None:
            metadata["children"] = "[]"
        else:
            for child in self._children:
                self.total_points += child.total_points
            populated_kids = [c.id for c in self._children if c.total_points > 0]
            metadata["children"] = json.dumps(populated_kids)

        unclean_path = self.filename.with_suffix(".needs_metadata.feather")
        if not unclean_path.exists():
           assert self.total_points == 0
           return 0
        self.data = feather.read_table(unclean_path)
        self.total_points += len(self.data)
        metadata["total_points"] = str(self.total_points)
        schema = pa.schema(self.schema, metadata = metadata)
        n_flushed = self.flush_data(self.filename, metadata, "uncompressed", schema)
        unclean_path.unlink()
        return self.total_points

    def flush_data(self, destination, metadata, compression, schema = None) -> int:
        """
        Flushes the locally stored data to disk

        Returns the number of points written.
        """
        if self.data is None:
            return 0

        if type(self.data) == pa.Table:
            frame = self.data.replace_schema_metadata(metadata)
        elif type(self.data) == list and type(self.data[0]) == pa.RecordBatch:
            if schema is None:
                schema = self.schema
            schema_copy = pa.schema(self.data[0].schema, metadata = metadata)
            frame = pa.Table.from_batches(self.data, schema_copy).combine_chunks()
        else:
            raise ValueError(f"Unrecognized data type: {type(self.data)}")
        if not "ix" in frame.column_names:
            raise TypeError("Should have ix already.")
        feather.write_feather(frame, destination, compression = compression)
        self.data = None
        return len(frame)

    @property
    def overflow_loc(self):
      return self.filename.with_suffix(".overflow.arrow")
    @property
    def overflow_buffer(self):
        if self._overflow_writer:
            return self._overflow_writer
        logger.debug(f"Opening overflow on {self.coords}")        
        if self.overflow_loc.exists():
            raise FileExistsError(f"Overflow file already exists: {self.overflow_loc}")
        self._sink = pa.OSFile(str(self.overflow_loc), 'wb')
        self._overflow_writer = \
            pa.ipc.new_file(self._sink, self.schema)
        return self._overflow_writer

    def partition_to_children(self, table) -> List[pa.Table]:
        # Coerce to a list in quadtree/octree order.
        frames = [table]
        pivot_dim : Tuple[str, float]
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
        return self.make_children()

    def make_children(self):
        # QUAD ONLY

        # Calling this forces child creation even when it's not wise.
        self._children = []
        midpoints = self.midpoints()

        for i in [0, 1]:
            xlim = [midpoints[0][1], self.extent['x'][i]]
            xlim.sort()
            for j in [0, 1]:
                ylim = [midpoints[1][1], self.extent['y'][j]]
                ylim.sort()
                extent = {"x": xlim, "y": ylim}
                coords = self.coords[0] + 1, self.coords[1]*2 + i, self.coords[2]*2 + j
                child = Tile(extent, coords, self.args)
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
        return "/".join(map(str,self.coords))
        
    def check_schema(self, table):
        if (self.schema is None):
            self.schema = table.schema
            return
        if not self.schema.equals(table.schema):
            logger.error("INSERTING:", table.schema)
            logger.error("OWN SCHEMA", self.schema)
            raise TypeError("Attempted to insert a table with a different schema.")

    def insert_table(self, table, tile_budget = float("Inf")):
        self.check_schema(table)
        insert_n_locally = self.TILE_SIZE - self.n_data_points
        if (insert_n_locally > 0):
            local_mask = np.zeros((table.shape[0]), bool)
            local_mask[:insert_n_locally] = True
            head = pc.filter(table, local_mask)
            if head.shape[0]:
                for batch in head.to_batches():
                    self.data.append(batch)
                self.n_data_points += head.shape[0]
            # All *unwritten* rows are now mapped into the main table to
            # flow into the children.
            table = table.filter(pc.invert(local_mask))
            if self.n_data_points == self.TILE_SIZE:
                # We've only just completed. Flush.
                self.first_flush()
        else:
            pass
        
        # 4, for quadtrees.
        children_per_tile = 2**(len(self.coords) - 1)
        # Always use previously created _overflow writer
        if not self._overflow_writer and (tile_budget >= children_per_tile or self._children is not None):
            # If we can afford to create children, do so.
            total_records = table.shape[0]
            if total_records == 0:
                return
            partitioning = self.partition_to_children(table)
            tiles_allowed_overflow = 0
            # The next block creates children and uses up some of the budget:
            # This one accounts for it.

            if self._children is None:
                tile_budget -= children_per_tile
                self.make_children()

            for child_tile, subset in zip(self.children, partitioning):
                # Each child gets a number of children proportional to its data share.
                # This works well on highly clumpy data.
                # Rebalance from (say) [3, 3, 3, 3]
                # to [0, 4, 4, 4] since anything less than four will lead to no kids.
                tiles_allowed = tile_budget * (subset.shape[0] / total_records) + tiles_allowed_overflow
                tiles_allowed_overflow = tiles_allowed % children_per_tile

                child_tile.insert_table(subset, tile_budget = tiles_allowed - tiles_allowed_overflow)
            return
        else:
            for batch in table.to_batches():
                self.overflow_buffer.write_batch(
                    batch
                )
def get_better_codes(col, counter = Counter()):
    for a in pc.value_counts(col):
        counter[a['values'].as_py()] += a['counts'].as_py()
    return counter

def get_recoding_arrays(files, col_name):
    countered = Counter()
    ingester = get_ingester(files, columns = [col_name])
    for batch in ingester:
        col = batch[col_name]
        countered = get_better_codes(col, countered)
    new_order = [pair[0] if pair[0] else "<NA>" for pair in countered.most_common(4094)]
    if len(new_order) == 4094:
        new_order.append("<Other>")
    new_order_dict = dict(zip(new_order, range(len(new_order))))
    return new_order, new_order_dict

def remap_dictionary(chunk, new_order_dict, new_order):
    # Switch a dictionary to use a pre-assigned set of keys. returns a new chunked dictionary array.
    index_map = pa.array([new_order_dict[str(k)] if str(k) in new_order_dict else len(new_order_dict) - 1 for k in chunk.dictionary], pa.uint16())
    if chunk.indices.null_count > 0:
        try:
            new_indices = pc.fill_null(pc.take(index_map, chunk.indices), new_order_dict["<NA>"])
        except KeyError:
            new_indices = pc.fill_null(pc.take(index_map, chunk.indices), new_order_dict["<Other>"])
    else:
        new_indices = pc.take(index_map, chunk.indices)

    return pa.DictionaryArray.from_arrays(new_indices, new_order)

def remap_all_dicts(col, new_order, new_order_dict):
    if (isinstance(col, pa.ChunkedArray)):
        return pa.chunked_array(remap_dictionary(chunk, new_order_dict, new_order) for chunk in col.chunks)
    else:
        return remap_dictionary(col, new_order_dict, new_order)


        
if __name__=="__main__":
    main()
