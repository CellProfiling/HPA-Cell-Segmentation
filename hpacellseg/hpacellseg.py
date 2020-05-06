"""HPA Cell Atlas Image Segmentation."""
import os
import imageio
import urllib.request
import numpy as np
from hpacellseg.cellsegmentator import CellSegmentator


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


class HPACellSeg:
    """HPA Cell Image Segmentation."""

    def __init__(
        self,
        image_channels,  # ['microtubules.png', 'er.png/None', 'nuclei.png'] or list
        nuclei_model="./nuclei_model.pth",
        cell_model="./cell_model.pth",
        batch_process=False,
    ):
        cell_channel, channel2nd, nuclei_channel = image_channels
        self.batch_process = batch_process
        if self.batch_process:
            assert isinstance(cell_channel, list)
            assert isinstance(nuclei_channel, list)
            assert len(cell_channel) == len(channel2nd) == len(nuclei_channel)
        else:
            assert isinstance(cell_channel, str)
            assert isinstance(nuclei_channel, str)
            cell_channel = [cell_channel]
            if channel2nd:
                assert isinstance(channel2nd, list)
                assert isinstance(channel2nd, str)
                channel2nd = [channel2nd]
            nuclei_channel = [nuclei_channel]
        cell_channel = [os.path.expanduser(item) for _, item in enumerate(cell_channel)]
        nuclei_channel = [
            os.path.expanduser(item) for _, item in enumerate(nuclei_channel)
        ]

        mt_data = list(map(lambda x: imageio.imread(x), cell_channel))
        nuclei_data = list(map(lambda x: imageio.imread(x), nuclei_channel))
        if channel2nd:
            channel2nd = [os.path.expanduser(item) for _, item in enumerate(channel2nd)]
            second_channel = list(map(lambda x: imageio.imread(x), channel2nd))
        else:
            second_channel = [
                np.zeros(item.shape, dtype=item.dtype) for _, item in enumerate(mt_data)
            ]
        self.cell_imgs = list(
            map(
                lambda item: np.dstack((item[0], item[1], item[2])),
                list(zip(mt_data, second_channel, nuclei_data)),
            )
        )

        if not os.path.exists(nuclei_model):
            os.makedirs(os.path.dirname(nuclei_model), exist_ok=True)
            print("Downloading nuclei segmentation model...")
            nuclei_model_url = (
                "https://kth.box.com/shared/static/l8z58wxkww9nn9syx9z90sclaga01mad.pth"
            )
            download_with_url(nuclei_model_url, nuclei_model)

        if not os.path.exists(cell_model):
            os.makedirs(os.path.dirname(cell_model), exist_ok=True)
            print("Downloading cell segmentation model...")
            if channel2nd: # place holder for 3channel model
                cell_model_url = (
                "https://kth.box.com/shared/static/hl2vuyi1lugywk6fr0drdz48w90gniyv.pth"
            )
            else:
                cell_model_url = (
                    "https://kth.box.com/shared/static/he8kbtpqdzm9xiznaospm15w4oqxp40f.pth"
                )
            download_with_url(cell_model_url, cell_model)
        self.nuclei_model = nuclei_model
        self.cell_model = cell_model

    def label_mask(self, scale_factor=0.5):
        seg = CellSegmentator(
            self.nuclei_model, self.cell_model, scale_factor=scale_factor, padding=True
        )
        cell_masks = seg.label_cells(self.cell_imgs)
        if self.batch_process:
            print(
                "The return value is list of cell mask data, following the cell_channel images"
            )
        else:
            cell_masks = cell_masks[0]
            print("Output the labeled cell mask")
        return cell_masks
