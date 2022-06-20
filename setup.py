from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='quadfeather',
      entry_points={
          'console_scripts': [
              'quadfeather = quadfeather.tiler:main',
              'quadfeather-test-data = quadfeather.demo:main'
          ],
      },
      packages=["quadfeather"],
      version='1.1.1,
      description="Quadtree tiling from CSV/Apache Arrow.",
      long_description = long_description,
      long_description_content_type = "text/markdown",
      url="http://github.com/bmschmidt/quadfeather",
      author="Benjamin Schmidt",
      author_email="bmschmidt@gmail.com",
      license="MIT",
      install_requires=["pyarrow", "numpy", "pandas"],
      tests_require=["pytest"], 
)
