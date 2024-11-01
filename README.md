# Quadfeather

Quickly create deep quadtree structures from CSV or feather files to allow serving over HTTP or quicker access to random portions of spatial data.

Each node in the quadtree is a single feather file containing some number of rows (default 2**16) and arrow 
metadata indicating the existence of children.


From the command line (basic):

WORK IN PROGRESS BELOW.

```bash
python -m pip install git+https://github.com/bmschmidt/quadfeather
quadfeather-test-data 255000 # populates a csv at tmp.csv with 255000 items
quadfeather --files tmp.csv --destination tiles
```

From a python iterable of Arrow record batches (advanced)

```python


```

## Todo

1. Allow splitting off of some columns.
2. Explain JS integration.
3. Octtree for 3-d data.

### Optional todo

1. Other partition strategies? Quadtree is easy to understand.,
2. Linked-open data standards.
