"""Set up file for HPA-Cell-Segmentation package."""
from pathlib import Path
from setuptools import find_packages, setup

PROJECT_DIR = Path(__file__).parent.resolve()
README_FILE = PROJECT_DIR / "README.org"
LONG_DESCR = README_FILE.read_text(encoding="utf-8")
VERSION = (PROJECT_DIR / "hpacellseg" / "VERSION").read_text().strip()
GITHUB_URL = "https://github.com/CellProfiling/HPA-Cell-Segmentation"
DOWNLOAD_URL = f"{GITHUB_URL}/archive/master.zip"

requirements = []
try:
    with open("requirements.txt", "r") as fd:
        requirements = [l.strip() for l in fd.readlines()]
except FileNotFoundError:
    print("WARNING: missing requirements.txt.")

requirements.append(
    "pytorch_zoo@https://github.com/haoxusci/pytorch_zoo/archive/master.zip"
)

setup(
    name="hpacellseg",
    version=VERSION,
    description="HPA Cell Segmentation",
    long_description=LONG_DESCR,
    long_description_content_type="text/markdown",
    author="Hao Xu",
    author_email="hao.xu@scilifelab.se",
    url=GITHUB_URL,
    download_url=DOWNLOAD_URL,
    license="Apache-2.0",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    python_requires=">=3.6",
    dependency_links=[],
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
)
