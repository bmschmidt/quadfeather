from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="quadfeather",
    entry_points={
        "console_scripts": [
            "quadfeather = quadfeather.tiler:cli",
            "quadfeather-test-data = quadfeather.demo:main",
            "quadfeather-sidecars = quadfeather.add_sidecars:add_sidecars_cli",
        ],
    },
    packages=["quadfeather"],
    version="2.0.0-RC1",
    description="Quadtree tiling from CSV/Apache Arrow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/bmschmidt/quadfeather",
    author="Benjamin Schmidt",
    author_email="bmschmidt@gmail.com",
    license="MIT",
    install_requires=["pyarrow", "numpy", "pandas"],
    tests_require=["pytest"],
)
