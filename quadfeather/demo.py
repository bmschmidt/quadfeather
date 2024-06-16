import numpy as np
from numpy import random
from random import choice, shuffle
import pandas as pd
import pyarrow as pa
import sys
from pyarrow import parquet as pq
from quadfeather.tiler import Rectangle
from pyarrow import compute as pc

dates = []

# generate some iso dates that may lack a month field
for y in range(1900, 2020):
    dates.append(f"{y}")
    for m in range(1, 13):
        dates.append(f"{y}-{m:02d}")
        for d in range(1, 6):
            dates.append(f"{y}-{m:02d}-{d:02d}")


def rbatch(SIZE, extent: Rectangle = Rectangle(x=(0, 100), y=(0, 100))):
    SIZE = int(SIZE)
    frames = []
    classes = ["Banana", "Strawberry", "Apple", "Mulberry"]
    for c in classes:
        x = random.lognormal(3.5, 0.4, size=SIZE // 4) * 1.5
        while len(x) < SIZE // 4:
            x = np.concatenate([x, random.lognormal(3.5, 0.4, size=SIZE // 4) * 1.5])
        x = x[: SIZE // 4]
        y = random.lognormal(3.5, 0.4, size=SIZE // 4) * 1.5
        while len(y) < SIZE // 4:
            y = np.concatenate([y, random.lognormal(3.5, 0.4, size=SIZE // 4) * 1.5])
        y = y[: SIZE // 4]

        date = random.choice(dates, size=len(x), replace=True)
        frame = pa.table(
            {
                "x": x,
                "y": y,
                "class": [c] * len(x),
                "quantity": random.random(len(x)),
                "date": date,
            }
        )
        frames.append(frame)
    return pa.concat_tables(frames)


def demo_parquet(
    path,
    size,
    batchsize=2e5,
    extent: Rectangle = Rectangle(x=(-100, 100), y=(-100, 100)),
):
    writer = None
    written = 0
    while written < size:
        if size - written < batchsize:
            batchsize = size - written
        batch = rbatch(batchsize, extent=extent)
        if writer is None:
            writer = pq.ParquetWriter(path, batch.schema)
        writer.write_table(batch)
        written = written + batchsize


def main(path="tmp.csv", SIZE=None):
    if SIZE is None:
        try:
            SIZE = int(sys.argv[1])
        except:
            SIZE = 100_000
    frames = rbatch(SIZE).to_pandas()
    frames = frames.sample(frac=1)
    # Add an unseen level at the very end.
    frames.iloc[-1, -1] = "2040-01-01"
    frames.to_csv(path, index=False)
