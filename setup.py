import os
from codecs import open
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

package = "lgl"

about = {}
with open(os.path.join(here, package, "__about__.py"), "r", "utf-8") as f:
    exec(f.read(), about)

def readme():
    with open(os.path.join(here, 'README.md'), 'r', 'utf-8') as f:
        return f.read()

requirements = ['torch',
                'numpy',
                'tqdm']


setup(name=about["__title__"],
      version=about["__version__"],
      description=about["__description__"],
      long_description=readme(),
      long_description_content_type='text/markdown',
      url=about["__url__"],
      author=about["__author__"],
      author_email=about["__author_email__"],
      packages=[package],
    #   package_data={'': ['s2_mgrs_grid.txt']},
    #   include_package_data=True,
      install_requires=requirements,
    #   extras_require={
    #     "gcp": ["google-auth", "google-cloud-bigquery", "pandas"],
    #     "planet": ["area", "planet", "rpcm"],
    #     "sentinelhub": ["sentinelhub"]
    #   },
      python_requires=">=3.7")