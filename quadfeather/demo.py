import numpy as np
from numpy import random
from random import choice, shuffle
import pyarrow as pa
import sys
from pyarrow import parquet as pq
from quadfeather.tiler import Rectangle
from pyarrow import compute as pc
try:
    import pandas as pd
except ImportError:
    pd = None

dates = []

# generate some iso dates that may lack a month field
for y in range(1900, 2020):
    dates.append(f"{y}")
    for m in range(1, 13):
        dates.append(f"{y}-{m:02d}")
        for d in range(1, 6):
            dates.append(f"{y}-{m:02d}-{d:02d}")

def rbatch(
    SIZE,
    extent: Rectangle = Rectangle(x=(0, 100), y=(0, 100)),
    method: str = "normal",
):
    SIZE = int(SIZE)
    frames = []
    if method == "lognormal":
        f = random.lognormal
        classes = [
            ("Banana", [5, 0.01], [5, 0.01]),
            ("Strawberry", [3.5, 0.4], [3, 0.3]),
            ("Apple", [4.6, 0.2], [3, 0.2]),
            ("Mulberry", [5.6, 0.6], [6, 0.5]),
        ]
    else:
        f = random.normal
        classes = [
            ("Banana", [0, 0.2], [0, 0.3]),
            ("Strawberry", [3, 0.05], [-3, 2]),
            ("Apple", [-4.6, 0.1], [-5, 0.25]),
            ("Mulberry", [5.6, 2.6], [6, 2.5]),
        ]

    for c, xparam, yparam in classes:
        x = f(*xparam, size=SIZE // 4) * 1.5
        x2 = f(*xparam, size=SIZE // 4) * 1.5
        while len(x) < SIZE // 4:
            x = np.concatenate([x, f(*xparam, size=SIZE // 4)])
            x2 =  np.concatenate([x2, f(*xparam, size=SIZE // 4)])
        x = x[: SIZE // 4]
        x2 = x2[: SIZE // 4]
        y = f(*yparam, size=SIZE // 4)
        y2 = f(*yparam, size=SIZE // 4)
        while len(y) < SIZE // 4:
            y = np.concatenate([y, f(*yparam, size=SIZE // 4)])
            y2 = np.concatenate([y2, f(*yparam, size=SIZE // 4)])
        y2 = y2[: SIZE // 4]

        date = random.choice(dates, size=len(x), replace=True)
        
        position = pa.StructArray.from_arrays([x2, y2], ['x', 'y'])
        frame = pa.table(
            {
                "x": x,
                "y": y,
                "position": position,
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
    if pd is None:
        raise ImportError("Must install pandas for demo script.")
    frames = rbatch(SIZE).to_pandas()
    frames = frames.sample(frac=1)
    # Add an unseen level at the very end.
    frames.iloc[-1, -1] = "2040-01-01"
    frames.to_csv(path, index=False)
