import pyarrow as pa
from pyarrow import parquet as pq, feather, json as pajson, csv
from pathlib import Path

class Ingester:
  def __init__(self, files : list[Path], batch_size : int = 1024 * 1024 * 1024, columns = None):
    # Allow iteration over a st
    # queue_size: maximum bytes in an insert chunk.

    self.files = files
    self.queue = []
    self.columns = columns
    self.batch_size = batch_size

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

  def batches(self):
    for file in self.files:
      feath = feather.read_table(file, columns=self.columns)
      for batch in feath.to_batches():
        yield batch 

class ParquetIngester(Ingester):
  def batches(self):
    for f in self.files:
      f = pq.ParquetFile(f)
      for batch in f.iter_batches(columns=self.columns):
        yield batch

def get_ingester(files : list[Path]) -> Ingester:
  assert len(set([f.suffix for f in files])) == 1, "All files must be of the same type"
  if files[0].suffix == ".parquet":
    return ParquetIngester(files)
  elif files[0].suffix == ".feather":
    return ArrowIngester(files)
  else:
    raise Exception("Unsupported file type", files[0].suffix)