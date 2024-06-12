import pytest

from quadfeather.tiler import *
from quadfeather.demo import main as demo_main, demo_parquet
import random
import json
import math
from pyarrow import parquet as pq, feather


class TestCSV:
    def test_basic(self, tmp_path):
        # Create a CSV file with four rows.
        csv_path = tmp_path / "test.csv"
        with csv_path.open("w") as fout:
            fout.write(
                """x,y,z
  1,1,1
  2,2,2
  3,3,3
  4,4,4"""
            )
        main(["--files", str(csv_path), "--destination", str(tmp_path / "tiles")])
        tb = feather.read_table(tmp_path / "tiles" / "0/0/0.feather")
        # Should introduce a new 'ix' column.
        for k in ["ix", "x", "y", "z"]:
            assert k in tb.column_names

    def test_failure_with_no_x_field(self, tmp_path):
        # Create a CSV= file with four rows.
        with pytest.raises(KeyError):
            csv_path = tmp_path / "test.csv"
            with csv_path.open("w") as fout:
                fout.write(
                    """r,y,z
          1,1,1
          2,2,2
          3,3,3
          4,4,4"""
                )
            main(["--files", str(csv_path), "--destination", str(tmp_path / "tiles")])

    def test_demo_file(self, tmp_path):
        demo_main(tmp_path / "test.csv")
        main(
            [
                "--files",
                str(tmp_path / "test.csv"),
                "--destination",
                str(tmp_path / "tiles"),
            ]
        )

    def test_demo_date_as_str(self, tmp_path):
        demo_main(tmp_path / "test.csv")
        main(
            [
                "--files",
                str(tmp_path / "test.csv"),
                "--destination",
                str(tmp_path / "tiles"),
                "--dtypes",
                "date=string",
            ]
        )

    def test_demo_date_as_str_small_block(self, tmp_path):
        demo_main(tmp_path / "test.csv", SIZE=100_000)
        main(
            [
                "--files",
                str(tmp_path / "test.csv"),
                "--destination",
                str(tmp_path / "tiles"),
                "--dtypes",
                "date=string",
            ],
            csv_block_size=4096,
        )
        root = str(tmp_path / "tiles")
        length = int(
            feather.read_table(tmp_path / "tiles" / "0/0/0.feather")
            .schema.metadata[b"total_points"]
            .decode("utf-8")
        )
        assert length == 100_000

    def test_if_break_categorical_chunks(self, tmp_path):
        input = tmp_path / "test.csv"
        with input.open("w") as fout:
            fout.write(f"x,y,cat\n")
            # Write 10,000 of each category to see if the later ones throw a dictionary error.
            for key in ["apple", "banana", "strawberry", "mulberry"]:
                for i in range(10000):
                    fout.write(f"{random.random()},{random.random()},{key}\n")

        main(
            [
                "--files",
                str(input),
                "--destination",
                str(tmp_path / "tiles"),
                "--dtypes",
                "cat=string",
            ],
            csv_block_size=1024,
        )

    def test_small_block_overflow(self, tmp_path):
        """
        We need to ensure that overflow works well. Usually we write relatively large blocks,
        so in this test we'll write extremely small ones and validate that everything gets inserted.
        """
        demo_main(tmp_path / "test.csv", SIZE=1_000_000)
        # This should require over 1,000 blocks.
        main(
            [
                "--files",
                str(tmp_path / "test.csv"),
                "--destination",
                str(tmp_path / "tiles"),
                "--tile_size",
                "1000",
            ]
        )
        root = str(tmp_path / "tiles")
        length = int(
            feather.read_table(tmp_path / "tiles" / "0/0/0.feather")
            .schema.metadata[b"total_points"]
            .decode("utf-8")
        )
        assert length == 1_000_000


class TestParquet:
    def test_big_parquet(self, tmp_path):
        size = 5_000_000
        demo_parquet(tmp_path / "test.parquet", size=size)
        main(
            [
                "--files",
                str(tmp_path / "test.parquet"),
                "--destination",
                str(tmp_path / "tiles"),
                "--tile_size",
                "5000",
            ]
        )
        length = int(
            feather.read_table(tmp_path / "tiles" / "0/0/0.feather")
            .schema.metadata[b"total_points"]
            .decode("utf-8")
        )
        assert length == size
        tb = feather.read_table(tmp_path / "tiles" / "0/0/0.feather")
        ps = tb["ix"].to_pylist()
        assert ps[0] == 0
        assert ps[1] == 1
        assert ps[-1] == 999


class TestStreaming:
    def test_streaming_batches(self, tmp_path):
        size = 5_000_000
        demo_parquet(
            tmp_path / "t.parquet", size=size, extent=Rectangle(x=(0, 100), y=(0, 100))
        )
        root_tile = Tile(
            extent=((0, 100), (0, 100)),
            writer_budget=32,
            basedir=tmp_path / "tiles",
            tile_size=1_000,
            first_tile_size=100,
        )

        for batch in pq.ParquetFile(tmp_path / "t.parquet").iter_batches(10_000):
            root_tile.insert(batch)
        manifest = root_tile.finalize()

        with open("manifest.json", "w") as fout:
            json.dump(manifest.to_dict(), fout)
