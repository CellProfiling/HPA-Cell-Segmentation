import os
import zipfile

import numpy as np
import imageio
import urllib.request
from hpacellseg.cellsegmentator import CellSegmentator


def download_with_url(url_string, file_path, unzip=False):
    with urllib.request.urlopen(url_string) as response, open(file_path, 'wb') as out_file:
        data = response.read() # a `bytes` object
        out_file.write(data)

    if unzip:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(file_path))

class HPACellSeg:
    def __init__(self, cell_channel, nuclei_channel, nuclei_model='./nuclei_model.pth', cell_model='./cell_model.pth', batch_process=False):
        self.batch_process = batch_process
        if self.batch_process:
            assert isinstance(cell_channel, list)
            assert isinstance(nuclei_channel, list)
            assert len(cell_channel)==len(nuclei_channel)
        else:
            assert isinstance(cell_channel, str)
            assert isinstance(nuclei_channel, str)
            cell_channel = [cell_channel]
            nuclei_channel = [nuclei_channel]
        cell_channel = [os.path.expanduser(item) for _, item in enumerate(cell_channel)]
        nuclei_channel = [os.path.expanduser(item) for _, item in enumerate(nuclei_channel)]
        
        self.cell_channel = cell_channel
        self.nuclei_channel = nuclei_channel

        mt_data = list(map(lambda x: imageio.imread(x), self.cell_channel))
        nuclei_data = list(map(lambda x: imageio.imread(x), self.nuclei_channel))
        empty_channel = [np.zeros(item.shape, dtype=item.dtype) for _, item in enumerate(mt_data)]
        self.cell_imgs = list(map(lambda item: np.dstack((item[0], item[1], item[2])), list(zip(mt_data, empty_channel, nuclei_data))))

        if not os.path.exists(nuclei_model):
            os.makedirs(os.path.dirname(nuclei_model),exist_ok=True)
            print('Downloading nuclei segmentation model...')
            nuclei_model_url = "https://kth.box.com/shared/static/l8z58wxkww9nn9syx9z90sclaga01mad.pth"
            download_with_url(nuclei_model_url, nuclei_model)

        if not os.path.exists(cell_model):
            os.makedirs(os.path.dirname(cell_model),exist_ok=True)
            print('Downloading cell segmentation model...')
            cell_model_url = "https://kth.box.com/shared/static/he8kbtpqdzm9xiznaospm15w4oqxp40f.pth"
            download_with_url(cell_model_url, cell_model)
        self.nuclei_model = nuclei_model
        self.cell_model = cell_model

    
    def label_mask(self):
        seg = CellSegmentator(self.nuclei_model, self.cell_model, scale_factor=0.5, padding=True)
        cell_masks = seg.label_cells(self.cell_imgs)
        if self.batch_process:
           print('The return value is list of cell mask data, following the cell_channel images')
        else:
            cell_masks = cell_masks[0]
            print('Output the labeled cell mask')
        return cell_masks 