"""Package for loading and running the nuclei and cell segmentation models programmaticly."""
import os
import sys

import cv2
import imageio
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from skimage import transform, util

from hpacellseg.constants import (MULTI_CHANNEL_CELL_MODEL_URL,
                                  NUCLEI_MODEL_URL, TWO_CHANNEL_CELL_MODEL_URL)
from hpacellseg.utils import download_with_url

NORMALIZE = {"mean": [124 / 255, 117 / 255, 104 / 255], "std": [1 / (0.0167 * 255)] * 3}


class CellSegmentator(object):
    """Uses pretrained DPN-Unet models to segment cells from images."""

    def __init__(
        self,
        nuclei_model="./nuclei_model.pth",
        cell_model="./cell_model.pth",
        scale_factor=0.25,
        device="cuda",
        padding=False,
        multi_channel_model=True,
    ):
        """Class for segmenting nuclei and whole cells from confocal microscopy images.

        It takes lists of images and returns the raw output from the
        specified segmentation model. Models can be automatically
        downloaded if they are not already available on the system.

        When working with images from the Huan Protein Cell atlas, the
        outputs from this class' methods are well combined with the
        label functions in the utils module.

        Note that for cell segmentation, there are two possible models
        available. One that works with 2 channeled images and one that
        takes 3 channels.

        Keyword arguments:
        nuclei_model -- A loaded torch nuclei segmentation model or the
                        path to a file which contains such a model.
                        If the argument is a path that points to a non-existant file,
                        a pretrained nuclei_model is going to get downloaded to the
                        specified path (default: './nuclei_model.pth').
        cell_model -- A loaded torch cell segmentation model or the
                      path to a file which contains such a model.
                      The cell_model argument can be None if only nuclei
                      are to be segmented (default: './cell_model.pth').
        scale_factor -- How much to scale images before they are fed to
                        segmentation models. Segmentations will be scaled back
                        up by 1/scale_factor to match the original image
                        (default: 0.25).
        device -- The device on which to run the models.
                  This should either be 'cpu' or 'cuda' or pointed cuda
                  device like 'cuda:0' (default: 'cuda').
        padding -- Whether to add padding to the images before feeding the
                   images to the network. (default: False).
        multi_channel_model -- Control whether to use the 3-channel cell model or not.
                               If True, use the 3-channel model, otherwise use the
                               2-channel version (default: True).
        """
        if device != "cuda" and device != "cpu" and "cuda" not in device:
            raise ValueError(f"{device} is not a valid device (cuda/cpu)")
        if device != "cpu":
            try:
                assert torch.cuda.is_available()
            except AssertionError:
                print("No GPU found, using CPU.", file=sys.stderr)
                device = "cpu"
        self.device = device

        if isinstance(nuclei_model, str):
            if not os.path.exists(nuclei_model):
                print(
                    f"Could not find {nuclei_model}. Downloading it now",
                    file=sys.stderr,
                )
                download_with_url(NUCLEI_MODEL_URL, nuclei_model)
            nuclei_model = torch.load(
                nuclei_model, map_location=torch.device(self.device)
            )
        if isinstance(nuclei_model, torch.nn.DataParallel) and device == "cpu":
            nuclei_model = nuclei_model.module

        self.nuclei_model = nuclei_model.to(self.device)

        self.multi_channel_model = multi_channel_model
        if isinstance(cell_model, str):
            if not os.path.exists(cell_model):
                print(
                    f"Could not find {cell_model}. Downloading it now", file=sys.stderr
                )
                if self.multi_channel_model:
                    download_with_url(MULTI_CHANNEL_CELL_MODEL_URL, cell_model)
                else:
                    download_with_url(TWO_CHANNEL_CELL_MODEL_URL, cell_model)
            cell_model = torch.load(cell_model, map_location=torch.device(self.device))
        self.cell_model = cell_model.to(self.device)
        self.scale_factor = scale_factor
        self.padding = padding

    def _image_conversion(self, images):
        """Convert/Format images to RGB image arrays list for cell predictions.

        Intended for internal use only.

        Keyword arguments:
        images -- list of lists of image paths/arrays. It should following the
                 pattern if with er channel input,
                 [
                     [microtubule_path0/image_array0, microtubule_path1/image_array1, ...],
                     [er_path0/image_array0, er_path1/image_array1, ...],
                     [nuclei_path0/image_array0, nuclei_path1/image_array1, ...]
                 ]
                 or if without er input,
                 [
                     [microtubule_path0/image_array0, microtubule_path1/image_array1, ...],
                     None,
                     [nuclei_path0/image_array0, nuclei_path1/image_array1, ...]
                 ]
        """
        microtubule_imgs, er_imgs, nuclei_imgs = images
        if self.multi_channel_model:
            if er_imgs is None:
                raise ValueError("Please specify the image path(s) for er channels!")

            assert isinstance(er_imgs, list)
        else:
            if not er_imgs is None:
                raise ValueError(
                    "second channel should be None for two channel model predition!"
                )

        if not isinstance(microtubule_imgs, list):
            raise ValueError("The microtubule images should be a list")
        if not isinstance(nuclei_imgs, list):
            raise ValueError("The microtubule images should be a list")

        if er_imgs:
            if not len(microtubule_imgs) == len(er_imgs) == len(nuclei_imgs):
                raise ValueError("The lists of images needs to be the same length")
        else:
            if not len(microtubule_imgs) == len(nuclei_imgs):
                raise ValueError("The lists of images needs to be the same length")

        if not all(isinstance(item, np.ndarray) for item in microtubule_imgs):
            microtubule_imgs = [
                os.path.expanduser(item) for _, item in enumerate(microtubule_imgs)
            ]
            nuclei_imgs = [
                os.path.expanduser(item) for _, item in enumerate(nuclei_imgs)
            ]

            microtubule_imgs = list(
                map(lambda item: imageio.imread(item), microtubule_imgs)
            )
            nuclei_imgs = list(map(lambda item: imageio.imread(item), nuclei_imgs))
            if er_imgs:
                er_imgs = [os.path.expanduser(item) for _, item in enumerate(er_imgs)]
                er_imgs = list(map(lambda item: imageio.imread(item), er_imgs))

        if not er_imgs:
            er_imgs = [
                np.zeros(item.shape, dtype=item.dtype)
                for _, item in enumerate(microtubule_imgs)
            ]
        cell_imgs = list(
            map(
                lambda item: np.dstack((item[0], item[1], item[2])),
                list(zip(microtubule_imgs, er_imgs, nuclei_imgs)),
            )
        )

        return cell_imgs

    def pred_nuclei(self, images):
        """Predict the nuclei segmentation.

        Keyword arguments:
        images -- A list of image arrays or a list of paths to images.
                  If as a list of image arrays, the images could be 2d images
                  of nuclei data array only, or must have the nuclei data in
                  the blue channel; If as a list of file paths, the images
                  could be RGB image files or gray scale nuclei image file
                  paths.

        Returns:
        predictions -- A list of predictions of nuclei segmentation for each nuclei image.
        """

        def _preprocess(image):
            if isinstance(image, str):
                image = imageio.imread(image)
            self.target_shape = image.shape
            if len(image.shape) == 2:
                image = np.dstack((image, image, image))
            image = transform.rescale(image, self.scale_factor, multichannel=True)
            nuc_image = np.dstack((image[..., 2], image[..., 2], image[..., 2]))
            if self.padding:
                rows, cols = nuc_image.shape[:2]
                self.scaled_shape = rows, cols
                nuc_image = cv2.copyMakeBorder(
                    nuc_image,
                    32,
                    (32 - rows % 32),
                    32,
                    (32 - cols % 32),
                    cv2.BORDER_REFLECT,
                )
            nuc_image = nuc_image.transpose([2, 0, 1])
            return nuc_image

        def _segment_helper(imgs):
            with torch.no_grad():
                mean = torch.as_tensor(NORMALIZE["mean"], device=self.device)
                std = torch.as_tensor(NORMALIZE["std"], device=self.device)
                imgs = torch.tensor(imgs).float()
                imgs = imgs.to(self.device)
                imgs = imgs.sub_(mean[:, None, None]).div_(std[:, None, None])

                imgs = self.nuclei_model(imgs)
                imgs = F.softmax(imgs, dim=1)
                return imgs

        preprocessed_imgs = map(_preprocess, images)
        predictions = map(lambda x: _segment_helper([x]), preprocessed_imgs)
        predictions = map(lambda x: x.to("cpu").numpy()[0], predictions)
        predictions = map(util.img_as_ubyte, predictions)
        predictions = list(map(self._restore_scaling_padding, predictions))
        return predictions

    def _restore_scaling_padding(self, n_prediction):
        """Restore an image from scaling and padding.

        This method is intended for internal use.
        It takes the output from the nuclei model as input.
        """
        n_prediction = n_prediction.transpose([1, 2, 0])
        if self.padding:
            n_prediction = n_prediction[
                32 : 32 + self.scaled_shape[0], 32 : 32 + self.scaled_shape[1], ...
            ]
        n_prediction[..., 0] = 0
        if not self.scale_factor == 1:
            n_prediction = cv2.resize(
                n_prediction,
                (self.target_shape[0], self.target_shape[1]),
                interpolation=cv2.INTER_AREA,
            )
        return n_prediction

    def pred_cells(self, images, precombined=False):
        """Predict the cell segmentation for a list of images.

        Keyword arguments:
        images -- list of lists of image paths/arrays. It should following the
                  pattern if with er channel input,
                  [
                      [microtubule_path0/image_array0, microtubule_path1/image_array1, ...],
                      [er_path0/image_array0, er_path1/image_array1, ...],
                      [nuclei_path0/image_array0, nuclei_path1/image_array1, ...]
                  ]
                  or if without er input,
                  [
                      [microtubule_path0/image_array0, microtubule_path1/image_array1, ...],
                      None,
                      [nuclei_path0/image_array0, nuclei_path1/image_array1, ...]
                  ]

                  The ER channel is required when multichannel is True
                  and required to be None when multichannel is False.

                  The images needs to be of the same size.
        precombined -- If precombined is True, the list of images is instead supposed to be
                       a list of RGB numpy arrays (default: False).

        Returns:
        predictions -- a list of predictions of cell segmentations.
        """

        def _preprocess(image):
            self.target_shape = image.shape
            if not len(image.shape) == 3:
                raise ValueError("image should has 3 channels")
            cell_image = transform.rescale(image, self.scale_factor, multichannel=True)
            if self.padding:
                rows, cols = cell_image.shape[:2]
                self.scaled_shape = rows, cols
                cell_image = cv2.copyMakeBorder(
                    cell_image,
                    32,
                    (32 - rows % 32),
                    32,
                    (32 - cols % 32),
                    cv2.BORDER_REFLECT,
                )
            cell_image = cell_image.transpose([2, 0, 1])
            return cell_image

        def _segment_helper(imgs):
            with torch.no_grad():
                mean = torch.as_tensor(NORMALIZE["mean"], device=self.device)
                std = torch.as_tensor(NORMALIZE["std"], device=self.device)
                imgs = torch.tensor(np.array(imgs)).float()
                imgs = imgs.to(self.device)
                imgs = imgs.sub_(mean[:, None, None]).div_(std[:, None, None])

                imgs = self.cell_model(imgs)
                imgs = F.softmax(imgs, dim=1)
                return imgs

        if not precombined:
            images = self._image_conversion(images)
        preprocessed_imgs = map(_preprocess, images)
        predictions = map(lambda x: _segment_helper([x]), preprocessed_imgs)
        predictions = map(lambda x: x.to("cpu").numpy()[0], predictions)
        predictions = map(self._restore_scaling_padding, predictions)
        predictions = list(map(util.img_as_ubyte, predictions))

        return predictions
