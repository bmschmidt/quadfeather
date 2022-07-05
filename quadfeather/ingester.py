import pyarrow as pa
from pyarrow import parquet as pq, feather, json as pajson, csv, ipc
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple, Set, Optional

class Ingester:
  def __init__(self, files : list[Path], batch_size : int = 1024 * 1024 * 1024, columns = None, destructive = False):
    # Allow iteration over a st
    # queue_size: maximum bytes in an insert chunk.
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

class ArrowIngester(Ingester):
  """
  The IPC format, not the feather format.
  """
  def batches(self):
    for file in self.files:
      source = pa.OSFile(str(file), 'rb')
      with pa.ipc.open_file(source, options = pa.ipc.IpcReadOptions(included_fields = self.columns)) as fin:
        for i in range(fin.num_record_batches):
          yield fin.get_batch(i)
      if self.destructive:
        file.unlink()
"""
class CSVIngester(Ingester):
"""

class FeatherIngester(Ingester):
  def batches(self):
    for file in self.files:
      fin = feather.read_table(file, columns = self.columns)
      for batch in fin.to_batches():
        yield batch
      if self.destructive:
        fin.unlink()
  
class ParquetIngester(Ingester):
  def batches(self):
    for f in self.files:
      f = pq.ParquetFile(f)
      for batch in f.iter_batches(columns=self.columns):
        yield batch
      if self.destructive:
        f.unlink()

def get_ingester(files : List[Path], destructive = False, columns : Optional[List[str]]= None) -> Ingester:
  assert len(set([f.suffix for f in files])) == 1, "All files must be of the same type"
  if files[0].suffix == ".parquet":
    return ParquetIngester(files, destructive = destructive, columns = columns)
  elif files[0].suffix == ".feather":
    return FeatherIngester(files, destructive = destructive, columns = columns)
  elif files[0].suffix == ".arrow":
    # This is the arrow IPC format, which can be slightly different (lacks a schema at the bottom.)
    return ArrowIngester(files, destructive = destructive, columns = columns)
  else:
    raise Exception("Unsupported file type", files[0].suffix)