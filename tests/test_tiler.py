import pytest

from quadfeather.tiler import *
from quadfeather.demo import main as demo_main, demo_parquet
import random
from pyarrow import parquet as pq, feather


def read_first_tile(q: Quadtree):
    return feather.read_table(q.basedir / "0/0/0.feather")


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
        main(
            **{
                "files": [str(csv_path)],
                "destination": str(tmp_path / "tiles"),
                "extent": None,
            }
        )
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
            main(
                **{
                    "files": [str(csv_path)],
                    "destination": str(tmp_path / "tiles"),
                    "extent": None,
                }
            )

    def test_demo_file(self, tmp_path):
        demo_main(tmp_path / "test.csv")
        main(
            **{
                "files": [str(tmp_path / "test.csv")],
                "destination": str(tmp_path / "tiles"),
                "extent": None,
            }
        )
        tb = feather.read_table(tmp_path / "tiles" / "manifest.feather")
        assert tb.num_rows > 4
        assert sum(tb["nPoints"].to_pylist()) == 100_000

    def test_demo_date_as_str_small_block(self, tmp_path):
        demo_main(tmp_path / "test.csv", SIZE=100_000)
        main(
            files=[tmp_path / "test.csv"],
            destination=tmp_path / "tiles",
            schema=pa.schema({"date": pa.string()}),
            csv_block_size=4096,
        )
        root = str(tmp_path / "tiles")
        t = feather.read_table(tmp_path / "tiles" / "manifest.feather")
        length = pc.sum(t["nPoints"]).as_py()
        assert length == 100_000

    def test_categorical_cast(self, tmp_path):
        input = tmp_path / "test.csv"
        with input.open("w") as fout:
            fout.write(f"x,y,number,cat\n")
            # Write 10,000 of each category to see if the later ones throw a dictionary error.
            rows = []
            for key, count in [
                ("apple", 10_000),
                ("banana", 1_000),
                ("strawberry", 100),
                ("mulberry", 10),
            ]:
                for _ in range(count):
                    rows.append(
                        f"{random.random()},{random.random()},{random.randint(0, 100)},{key}\n"
                    )
            random.shuffle(rows)
            for line in rows:
                fout.write(line)
        quadtree = main(
            files=[input],
            destination=tmp_path / "tiles",
            schema=pa.schema({"number": pa.int32()}),
            dictionaries={
                "cat": pa.array(["apple", "banana", "strawberry", "mulberry"])
            },
            sidecars={"cat": "cat"},
            csv_block_size=1024,
        )

        batches = []
        for fin in tmp_path.glob("**/*.cat.feather"):
            batches.append(feather.read_table(fin))

        alltogether = pa.concat_tables(batches)
        counts = alltogether["cat"].value_counts().to_pydict()

    def test_if_break_categorical_chunks(self, tmp_path):
        input = tmp_path / "test.csv"
        with input.open("w") as fout:
            fout.write(f"x,y,cat\n")
            # Write 10,000 of each category to see if the later ones throw a dictionary error.
            for key in ["apple", "banana", "strawberry", "mulberry"]:
                for i in range(10000):
                    fout.write(f"{random.random()},{random.random()},{key}\n")

        qtree = main(
            files=[input],
            destination=tmp_path / "tiles",
            dictionaries={
                "cat": pa.array(["apple", "banana", "strawberry", "mulberry"])
            },
            sidecars={"cat": "cat"},
            first_tile_size=1000,
            csv_block_size=1024,
        )
        t1 = qtree.read_root_table("")
        assert t1.num_rows == 1000
        assert "cat" not in t1.column_names
        t2 = qtree.read_root_table("cat")
        assert t2.num_rows == 1000
        assert "cat" in t2.column_names
        assert pa.types.is_dictionary(t2.schema.field_by_name("cat").type)

    def test_small_block_overflow(self, tmp_path):
        """
        We need to ensure that overflow works well. Usually we write relatively large blocks,
        so in this test we'll write extremely small ones and validate that everything gets inserted.
        """
        demo_main(tmp_path / "test.csv", SIZE=1_000_000)
        # This should require over 1,000 blocks.
        qtree = main(
            files=[tmp_path / "test.csv"],
            destination=tmp_path / "tiles",
            tile_size=1000,
            first_tile_size=100,
            sidecars={"ix": "ix"},
        )
        tb = feather.read_table(tmp_path / "tiles" / "manifest.feather")
        assert sum(tb["nPoints"].to_pylist()) == 1_000_000
        assert all([row <= 1000 for row in tb["nPoints"].to_pylist()])
        tb1 = qtree.read_root_table("")
        assert tb1.num_rows == 100

        tb2 = qtree.read_root_table("ix")
        assert "ix" in tb2.column_names


class TestParquet:
    def test_big_parquet(self, tmp_path):
        size = 5_000_000
        demo_parquet(tmp_path / "test.parquet", size=size)
        qtree = main(
            files=[tmp_path / "test.parquet"],
            destination=tmp_path / "tiles",
            tile_size=5000,
            first_tile_size=1000,
        )
        manifest = qtree.manifest_table
        assert pc.sum(manifest["nPoints"]).as_py() == size
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
        qtree = Quadtree(
            mode="write",
            extent=((0, 100), (0, 100)),
            max_open_filehandles=33,
            basedir=tmp_path / "tiles",
            tile_size=9_000,
            dictionaries={
                "cat": pa.array(["apple", "banana", "strawberry", "mulberry"]),
                "sidecars": {"cat": "cat"},
            },
            first_tile_size=1000,
        )

        for batch in pq.ParquetFile(tmp_path / "t.parquet").iter_batches(10_000):
            qtree.insert(batch)

        qtree.finalize()

        manifest = qtree.manifest_table

        assert pc.sum(manifest["nPoints"]).as_py() == size

        tb = feather.read_table(tmp_path / "tiles" / "0/0/0.feather")


class TestAppends:
    def test_build_index(self, tmp_path):
        insert_data = pa.table(
            {
                "id": pa.array(np.arange(10_000)).cast(pa.string()),
                "x": np.random.random(10_000),
                "y": np.random.random(10_000),
            }
        )
        qtree = Quadtree(
            mode="write",
            extent=((0, 1), (0, 1)),
            max_open_filehandles=33,
            basedir=tmp_path / "tiles",
            tile_size=100,
            first_tile_size=50,
        )

        qtree.insert(insert_data)

        qtree.finalize()

        qtree.build_bloom_index("id", None)

    def test_join_onfile(self, tmp_path):
        insert_data = pa.table(
            {
                "id": pa.array(np.arange(10_000)).cast(pa.string()),
                "x": np.random.random(10_000),
                "y": np.random.random(10_000),
            }
        )
        qtree = Quadtree(
            mode="write",
            extent=((0, 1), (0, 1)),
            max_open_filehandles=33,
            basedir=tmp_path / "tiles",
            tile_size=65_000,
            first_tile_size=65_000,
        )

        qtree.insert(insert_data)

        qtree.finalize()

        ids = pa.array(np.arange(10_000)).cast(pa.string())
        insert_tb = pa.table({"id": ids, "join_field": ids})
        # Shuffle the insert table.
        shuffled_indices = np.random.permutation(len(insert_tb))
        insert_tb = insert_tb.take(pa.array(shuffled_indices))

        def stream():
            start = 0
            while start < len(insert_tb):
                yield insert_tb.take(np.arange(start, start + 1000))
                start += 1000

        qtree.join(stream(), "id", "new_sidecar", None)
        m = qtree.read_root_table("new_sidecar")
        root_ids = qtree.read_root_table(None)["id"]
        assert not "id" in m.column_names
        assert "join_field" in m.column_names

        assert (pc.all(pc.equal(m["join_field"], root_ids))).as_py()

    def test_join(self, tmp_path):
        insert_data = pa.table(
            {
                "id": pa.array(np.arange(10_000)).cast(pa.string()),
                "x": np.random.random(10_000),
                "y": np.random.random(10_000),
            }
        )
        qtree = Quadtree(
            mode="write",
            extent=((0, 1), (0, 1)),
            max_open_filehandles=33,
            basedir=tmp_path / "tiles",
            tile_size=100,
            first_tile_size=50,
        )

        qtree.insert(insert_data)

        qtree.finalize()

        ids = pa.array(np.arange(10_000)).cast(pa.string())
        insert_tb = pa.table({"id": ids, "join_field": ids})
        # Shuffle the insert table.
        shuffled_indices = np.random.permutation(len(insert_tb))
        insert_tb = insert_tb.take(pa.array(shuffled_indices))

        def stream():
            start = 0
            while start < len(insert_tb):
                yield insert_tb.take(np.arange(start, start + 1000))
                start += 1000

        qtree.join(stream(), "id", "new_sidecar", None)
        m = qtree.read_root_table("new_sidecar")
        root_ids = qtree.read_root_table(None)["id"]
        assert not "id" in m.column_names
        assert "join_field" in m.column_names

        assert (pc.all(pc.equal(m["join_field"], root_ids))).as_py()


if __name__ == "__main__":
    logger = logging.getLogger("quadfeather")
    logger.setLevel(level=logging.DEBUG)
    # t = TestStreaming()
    # t.test_streaming_batches(tmp_path=Path("tmp"))
    t = TestAppends()
    t.test_join(tmp_path=Path("tmp"))
