import pytest

from quadfeather.tiler import *
from quadfeather.demo import main as demo_main

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
  
  @pytest.mark.xfail
  def test_failure_with_no_x_field(self, tmp_path):
    # Create a CSV file with four rows.
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
