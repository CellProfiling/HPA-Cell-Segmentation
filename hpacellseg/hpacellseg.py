"""HPA Cell Atlas Image Segmentation."""
import os

import imageio
import numpy as np
from hpacellseg.cellsegmentator import CellSegmentator

from hpacellseg.constants import *
from hpacellseg.utils import download_with_url


class HPACellSeg:
    """HPA Cell Image Segmentation."""

    def __init__(
        self,
        image_channels,  # ['microtubules.png', 'er.png/None', 'nuclei.png'] or list
        nuclei_model="./nuclei_model.pth",
        cell_model="./cell_model.pth",
        device='cuda',
        batch_process=False,
    ):
        self.device = device
        cell_channel, channel2nd, nuclei_channel = image_channels
        self.batch_process = batch_process
        if self.batch_process:
            assert isinstance(cell_channel, list)
            assert isinstance(nuclei_channel, list)
            if channel2nd:
                assert isinstance(channel2nd, list)
                assert len(cell_channel) == len(channel2nd) == len(nuclei_channel)
            else:
                assert len(cell_channel) == len(nuclei_channel)
        else:
            assert isinstance(cell_channel, str)
            assert isinstance(nuclei_channel, str)
            cell_channel = [cell_channel]
            if channel2nd:
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
            download_with_url(NUCLEI_MODEL_URL, nuclei_model)

        if not os.path.exists(cell_model):
            os.makedirs(os.path.dirname(cell_model), exist_ok=True)
            print("Downloading cell segmentation model...")
            if channel2nd:  # place holder for 3channel model
                cell_model_url = MULTI_CHANNEL_CELL_MODEL_URL
            else:
                cell_model_url = CELL_MODEL_URL
            download_with_url(CELL_MODEL_URL, cell_model)
        self.nuclei_model = nuclei_model
        self.cell_model = cell_model

    def label_mask(self, scale_factor=0.25):
        seg = CellSegmentator(
            self.nuclei_model,
            self.cell_model,
            scale_factor=scale_factor,
            device=self.device,
            padding=True
        )
        cell_masks = seg.label_cells(self.cell_imgs)
        if self.batch_process:
            print(
                "The return value is list of cell mask data, following the "
                "cell_channel images"
            )
        else:
            cell_masks = cell_masks[0]
            #print("Output the labeled cell mask")
        return cell_masks
