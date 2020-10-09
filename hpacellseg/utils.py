"""Utility functions for the HPA Cell Segmentation package."""
import os.path
import urllib
import zipfile


def download_with_url(url_string, file_path, unzip=False):
    """Download file with a link."""
    with urllib.request.urlopen(url_string) as response, open(
        file_path, "wb"
    ) as out_file:
        data = response.read()  # a `bytes` object
        out_file.write(data)

    if unzip:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(file_path))
