"""Package for loading and running the nuclei and cell segmentation models programmaticly."""
import os
import sys

import cv2
import imageio
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from skimage import util

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
            model_width_height=512,
            device="cuda",
            multi_channel_model=True,
            return_without_scale_restore=False
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
        model_width_height --
        device -- The device on which to run the models.
                  This should either be 'cpu' or 'cuda' or pointed cuda
                  device like 'cuda:0' (default: 'cuda').
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
        self.model_width_height = model_width_height
        self.return_without_scale_restore = return_without_scale_restore

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
            if not isinstance(er_imgs, list):
                raise ValueError("Please speicify the image path(s) for er channels!")
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

        def _preprocess(images):
            if isinstance(images[0], str):
                raise NotImplementedError('Currently the model requires images as numpy arrays, not paths.')
                # images = [imageio.imread(image_path) for image_path in images]
            self.target_shapes = [image.shape for image in images]
            images = [cv2.resize(image, (self.model_width_height, self.model_width_height))
                      if image.shape[0] != self.model_width_height or image.shape[
                1] != self.model_width_height else image
                      for image in images]

            nuc_images = np.array([np.dstack((image[..., 2], image[..., 2], image[..., 2])) if len(image.shape) >= 3
                                   else np.dstack((image, image, image)) for image in images])
            nuc_images = nuc_images.transpose([0, 3, 1, 2])
            return nuc_images

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

        preprocessed_imgs = _preprocess(images)
        predictions = _segment_helper(preprocessed_imgs)
        predictions = predictions.to("cpu").numpy()
        predictions = [self._restore_scaling(util.img_as_ubyte(pred), target_shape)
                       for pred, target_shape in zip(predictions, self.target_shapes)]
        return predictions

    def _restore_scaling(self, n_prediction, target_shape):
        """Restore an image from scaling and padding.

        This method is intended for internal use.
        It takes the output from the nuclei model as input.
        """
        n_prediction = n_prediction.transpose([1, 2, 0])
        n_prediction[..., 0] = 0
        if not self.return_without_scale_restore:
            n_prediction = cv2.resize(
                n_prediction,
                (target_shape[0], target_shape[1]),
                interpolation=cv2.INTER_NEAREST,
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

        def _preprocess(images):
            self.target_shapes = [image.shape for image in images]
            for image in images:
                if not len(image.shape) == 3:
                    raise ValueError("image should has 3 channels")
            images = np.array([cv2.resize(image, (self.model_width_height, self.model_width_height))
                               if image.shape[0] != self.model_width_height or image.shape[1] != self.model_width_height
                               else image
                               for image in images])
            cell_images = images.transpose([0, 3, 1, 2])
            return cell_images

        def _segment_helper(imgs):
            with torch.no_grad():
                mean = torch.as_tensor(NORMALIZE["mean"], device=self.device)
                std = torch.as_tensor(NORMALIZE["std"], device=self.device)
                imgs = torch.tensor(imgs).float()
                imgs = imgs.to(self.device)
                imgs = imgs.sub_(mean[:, None, None]).div_(std[:, None, None])
                imgs = self.cell_model(imgs)
                imgs = F.softmax(imgs, dim=1)
                return imgs

        if not precombined:
            images = self._image_conversion(images)
        preprocessed_imgs = _preprocess(images)
        predictions = _segment_helper(preprocessed_imgs)
        predictions = predictions.to("cpu").numpy()
        predictions = [self._restore_scaling(util.img_as_ubyte(pred), target_shape)
                       for pred, target_shape in zip(predictions, self.target_shapes)]
        return predictions
