import pytest

from quadfeather.tiler import *
from quadfeather.demo import main as demo_main
import random

class TestCSV():
  def test_basic(self, tmp_path):
    # Create a CSV file with four rows.
    csv_path = tmp_path / "test.csv"
    with csv_path.open("w") as fout:
      fout.write("""x,y,z
  1,1,1
  2,2,2
  3,3,3
  4,4,4""")
    main(["--files", str(csv_path), '--destination', str(tmp_path / "tiles")])
  
  def test_failure_with_no_x_field(self, tmp_path):
    # Create a CSV file with four rows.
    with pytest.raises(KeyError):
      csv_path = tmp_path / "test.csv"
      with csv_path.open("w") as fout:
        fout.write("""r,y,z
    1,1,1
    2,2,2
    3,3,3
    4,4,4""")
      main(["--files", str(csv_path), '--destination', str(tmp_path / "tiles")])

  def test_demo_file(self, tmp_path):
    demo_main(tmp_path / "test.csv")
    main(["--files", str(tmp_path / "test.csv"), '--destination', str(tmp_path / "tiles")])

  def test_demo_date_as_str(self, tmp_path):
    demo_main(tmp_path / "test.csv")
    main(["--files", str(tmp_path / "test.csv"), '--destination', str(tmp_path / "tiles"), '--dtypes', 'date=string'])

  def test_demo_date_as_str_small_block(self, tmp_path):
    demo_main(tmp_path / "test.csv")
    main(["--files", str(tmp_path / "test.csv"), '--destination', str(tmp_path / "tiles"), '--dtypes', 'date=string'], csv_block_size = 4096)


  def test_if_break_categorical_chunks(self, tmp_path):
    input = tmp_path / "test.csv"
    with input.open("w") as fout:
      fout.write(f"x,y,cat\n")
      # Write 10,000 of each category to see if the later ones throw a dictionary error.
      for key in ["apple", "banana", "strawberry", "mulberry"]:
        for i in range(10000):
          fout.write(f"{random.random()},{random.random()},{key}\n")

    main(["--files", str(input), '--destination', str(tmp_path / "tiles"),
     '--dtypes', 'cat=string'], csv_block_size = 1024)