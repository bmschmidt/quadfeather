import pyarrow as pa
from pyarrow import parquet as pq, feather, json as pajson, csv, ipc
from pathlib import Path
from typing import (
    DefaultDict,
    Dict,
    Union,
    List,
    Tuple,
    Set,
    Optional,
    Iterable,
    Callable,
)
from abc import ABC, abstractmethod


class Ingester(ABC):
    def __init__(
        self,
        files: Union[Path, List[Path]] = [],
        batch_size: int = 1024 * 1024 * 1024,
        columns: Optional[List[str]] = None,
        destructive: bool = False,
    ):
        # Allow iteration over a st
        # queue_size: maximum bytes in an insert chunk.
        if not isinstance(files, list):
            files = [files]
        self.files = files
        self.queue = []
        self.columns = columns
        self.batch_size = batch_size
        self.destructive = destructive

    def __iter__(self):
        queue_length = 0
        for batch in self.batches():
            self.queue.append(batch)
            queue_length = queue_length + batch.nbytes
            if queue_length > self.batch_size:
                tb = pa.Table.from_batches(batches=self.queue)
                yield tb.combine_chunks()
                self.queue = []
                queue_length = 0
        if len(self.queue) > 0:
            tb = pa.Table.from_batches(batches=self.queue)
            yield tb.combine_chunks()

    @abstractmethod
    def batches() -> Iterable[pa.RecordBatch]:
        pass


class BatchIngester(Ingester):
    """
    consumes a stream of arrow record batches and feeds them into a quadtree.
    """

    def __init__(self, batch_caller: Callable[[], Iterable[pa.RecordBatch]], **kwargs):
        self.batch_caller = batch_caller
        super().__init__(**kwargs)

    def batches(self):
        yield from self.batch_caller()


class CSVIngester(Ingester):
    def batches(self) -> Iterable[pa.RecordBatch]:
        for file in self.files:
            with open(file, "rb") as fin:
                reader = csv.open_csv(fin, read_options=csv.ReadOptions())
                for batch in reader:
                    yield batch
            if self.destructive:
                file.unlink()


class ArrowIngester(Ingester):
    """
    The IPC format, not the feather format.
    """

    def batches(self) -> Iterable[pa.RecordBatch]:
        for file in self.files:
            source = pa.OSFile(str(file), "rb")
            with pa.ipc.open_file(
                source, options=pa.ipc.IpcReadOptions(included_fields=self.columns)
            ) as fin:
                for i in range(fin.num_record_batches):
                    yield fin.get_batch(i)
            source.close()
            if self.destructive:
                file.unlink()


"""
class CSVIngester(Ingester):
"""


class FeatherIngester(Ingester):
    def batches(self) -> Iterable[pa.RecordBatch]:
        for file in self.files:
            fin = feather.read_table(file, columns=self.columns)
            for batch in fin.to_batches():
                yield batch
            if self.destructive:
                fin.unlink()


class ParquetIngester(Ingester):
    def batches(self):
        for fname in self.files:
            f = pq.ParquetFile(fname)
            for batch in f.iter_batches(columns=self.columns):
                yield batch
            if self.destructive:
                fname.unlink()


def get_ingester(
    files: Union[Path, List[Path]],
    destructive=False,
    columns: Optional[List[str]] = None,
) -> Ingester:
    if not isinstance(files, list):
        files = [files]
    assert (
        len(set([f.suffix for f in files])) == 1
    ), "All files must be of the same type"
    if files[0].suffix == ".csv":
        return CSVIngester(files, destructive=destructive, columns=columns)
    if files[0].suffix == ".parquet":
        return ParquetIngester(files, destructive=destructive, columns=columns)
    elif files[0].suffix == ".feather":
        return FeatherIngester(files, destructive=destructive, columns=columns)
    elif files[0].suffix == ".arrow":
        # This is the arrow IPC format, which can be slightly different (lacks a schema at the bottom.)
        return ArrowIngester(files, destructive=destructive, columns=columns)
    else:
        raise Exception("Unsupported file type", files[0].suffix)
