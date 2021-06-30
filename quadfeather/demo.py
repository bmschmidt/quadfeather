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
      scale = random.random()
      x = random.normal(loc=mid_x, scale=(scale + .5)/3, size=SIZE // 4)
      mid_y = np.random.normal()
      scale = random.random()
      y = random.normal(loc=mid_y, scale=(scale + .5)/3, size=SIZE // 4)
      frame = pd.DataFrame({"x": x, "y": y, "class": c, "quantity": random.random(SIZE // 4)})
      frames.append(frame)

  pd.concat(frames).sample(frac = 1).to_csv("tmp.csv", index = False)