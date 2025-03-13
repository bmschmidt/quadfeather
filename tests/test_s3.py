from pyarrow import fs
from quadfeather.tiler import *
from pathlib import Path

import pytest

def get_s3_filesystem(region: str = "us-east-2"):
    return fs.S3FileSystem(
        region=region,
    )

@pytest.mark.skip(reason="This is a slow test that requires a real S3 bucket.")
def test_s3_filesystem(bucket_name: str,NUM_POINTS=100_000, TILE_SIZE=10_000):
    fs = get_s3_filesystem()
    basedir = Path(bucket_name) / "tiles"
    print("Creating test data...")
    insert_data = pa.table(
        {
            "id": pa.array(np.arange(NUM_POINTS)).cast(pa.string()),
            "x": np.random.normal(0, 10, NUM_POINTS),
            "y": np.random.normal(0, 10, NUM_POINTS),
        }
    )
    xtent = pc.min_max(insert_data["x"]).as_py()
    ytent = pc.min_max(insert_data["y"]).as_py()
    print("Creating quadtree...")
    qtree = Quadtree(
        mode="write",
        extent=((xtent["min"], xtent["max"]), (ytent["min"], ytent["max"])),
        max_open_filehandles=33,
        basedir=basedir,
        tile_size=TILE_SIZE,
        first_tile_size=int(TILE_SIZE / 4),
        filesystem=fs,
    )
    print("Inserting data...")
    qtree.insert(insert_data)

    print("Finalizing quadtree...")
    qtree.finalize()

    print("Adding sidecar...")
    ids = pa.array(np.arange(NUM_POINTS)).cast(pa.string())
    insert_tb = pa.table({"id": ids, "join_field": ids})
    # Shuffle the insert table.
    shuffled_indices = np.random.permutation(len(insert_tb))
    insert_tb = insert_tb.take(pa.array(shuffled_indices))

    def stream():
        start = 0
        while start < len(insert_tb):
            yield insert_tb.take(np.arange(start, start + int(NUM_POINTS / 100)))
            start += int(NUM_POINTS / 100)

    qtree.join(stream(), "id", "new_sidecar")

    ### First, we just confirm that the root table was correctly built.
    m = qtree.read_root_table("new_sidecar")
    root_ids = qtree.read_root_table(None)["id"]
    assert not "id" in m.column_names
    assert "join_field" in m.column_names
    assert (pc.all(pc.equal(m["join_field"], root_ids))).as_py()


# Run with aws-vault exec
if __name__ == "__main__":
    test_s3_filesystem("quadfeather")
