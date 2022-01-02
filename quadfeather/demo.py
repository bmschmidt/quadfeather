import numpy as np
from numpy import random
from random import choice, shuffle
import pandas as pd
import sys

dates = []

# generate some iso dates that may lack a month field
for y in range(1900, 2020):
  dates.append(f"{y}")
  for m in range(1, 13):
    dates.append(f"{y}-{m:02d}")
    for d in range(1, 6):
      dates.append(f"{y}-{m:02d}-{d:02d}")

def main(path = "tmp.csv"):
  try:
    SIZE = int(sys.argv[1])
  except:
    SIZE = 100_000

  classes = ["Banana", "Strawberry", "Apple", "Mulberry"]
  frames = []
  for c in classes:
      mid_x = np.random.normal()
      scale = random.random()
      x = random.normal(loc=mid_x, scale=(scale + .5)/3, size=SIZE // 4)
      mid_y = np.random.normal()
      scale = random.random()
      y = random.normal(loc=mid_y, scale=(scale + .5)/3, size=SIZE // 4)
      date = random.choice(dates)
      frame = pd.DataFrame({
        "x": x, "y": y, "class": c,
        "quantity": random.random(SIZE // 4),
        "date": date
        })
      frames.append(frame)


  frames = pd.concat(frames).sample(frac = 1)
  # Add an unseen level at the very end.
  frames.iloc[-1, -1] = "2040-01-01"
  frames.to_csv(path, index = False)