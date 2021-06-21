import numpy as np
from numpy import random
import pandas as pd
import sys

def main():
  try:
    SIZE = int(sys.argv[1])
  except:
    SIZE = 100_000

  classes = ["Banana", "Strawberry", "Apple", "Mulberry"]
  frames = []
  for c in classes:
      mid_x = np.random.normal()
      x = random.normal(loc=mid_x, scale=.3, size=SIZE // 4)
      mid_y = np.random.normal()
      y = random.normal(loc=mid_y, scale=.3, size=SIZE // 4)
      frame = pd.DataFrame({"x": x, "y": y, "class": c, "size": random.random(SIZE // 4)})
      frames.append(frame)
  pd.concat(frames).sample(frac = 1).to_csv("/tmp/tmp.csv", index = False)



  try:
    SIZE = int(sys.argv[1])
  except:
    SIZE = 100_000

  classes = ["Banana", "Strawberry", "Apple", "Mulberry"]
  frames = []
  for c in classes:
      mid_x = np.random.normal()
      x = random.normal(loc=mid_x, scale=.3, size=SIZE // 4)
      mid_y = np.random.normal()
      y = random.normal(loc=mid_y, scale=.3, size=SIZE // 4)
      frame = pd.DataFrame({"x": x, "y": y, "class": c, "size": random.random(SIZE // 4)})
      frames.append(frame)

  pd.concat(frames).sample(frac = 1).to_csv("tmp.csv", index = False)