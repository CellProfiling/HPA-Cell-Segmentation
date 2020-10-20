"""Package for loading and running the nuclei and cell segmentation models programmaticly."""
import os
import sys

import cv2
import imageio
import numpy as np
import scipy.ndimage as ndi
from skimage import measure, transform, segmentation, filters, util
from skimage.morphology import (
    closing,
    disk,
    remove_small_holes,
    remove_small_objects,
)
import torch
import torch.nn
import torch.nn.functional as F
from hpacellseg.constants import (
    MULTI_CHANNEL_CELL_MODEL_URL,
    TWO_CHANNEL_CELL_MODEL_URL,
    NUCLEI_MODEL_URL,
)
from hpacellseg.utils import download_with_url


NORMALIZE = {
    "mean": [124 / 255, 117 / 255, 104 / 255],
    "std": [1 / (0.0167 * 255)] * 3,
}


class CellSegmentator(object):
    """Uses pretrained DPN-Unet models to segment cells from images."""

    def __init__(
        self,
        nuclei_model='./nuclei_model.pth',
        cell_model='./cell_model.pth',
        scale_factor=0.25,
        device="cuda",
        padding=False,
        batch_process=False,
        multi_channel_model=True,
        post_processing=True
    ):
        """
        Keyword arguments:
        nuclei_model -- A loaded torch nuclei segmentation model or the
                     path to a file which contains such a model.
        cell_model -- A loaded torch cell segmentation model or the
                     path to a file which contains such a model.
                     The cell segmentator argument can be None if only nucleli
                     are to be segmented. (default: None)
        scale_factor -- How much to scale images before they are fed to
                     segmentation models. Segmentations will be scaled back
                     up by 1/scale_factor to match the original image
                     (default: 1.0).
        device -- The device on which to run the models.
                 This should either be 'cpu' or 'cuda' or pointed cuda
                 device like 'cuda:0' (default: 'cuda').
        padding -- Whether to add padding to the images before feeding the
                 images to the network. (default: False)
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
                    f"Could not find {cell_model}. Downloading it now",
                    file=sys.stderr,
                )
                if self.multi_channel_model:
                    download_with_url(MULTI_CHANNEL_CELL_MODEL_URL, cell_model)
                else:
                    download_with_url(TWO_CHANNEL_CELL_MODEL_URL, cell_model)
            cell_model = torch.load(
                cell_model, map_location=torch.device(self.device)
            )
        # if isinstance(cell_model, torch.nn.DataParallel) and device == 'cpu':
        #    cell_model = cell_model.module
        self.cell_model = cell_model.to(self.device)
        self.batch_process = batch_process
        self.scale_factor = scale_factor
        self.padding = padding
        self.post_processing = post_processing

    def batch_check(self, images):
        microtubule_images, er_images, nuclei_images = images
        if self.multi_channel_model:
            assert (
                er_images is not None
            ), "Please speicify the image path(s) for er channels!"
        if self.batch_process:
            assert isinstance(microtubule_images, list)
            assert isinstance(nuclei_images, list)
            if er_images:
                assert isinstance(er_images, list)
                assert (
                    len(microtubule_images)
                    == len(er_images)
                    == len(nuclei_images)
                )
            else:
                assert len(microtubule_images) == len(nuclei_images)
        else:
            assert isinstance(microtubule_images, str) #raise error
            assert isinstance(nuclei_images, str)
            microtubule_images = [microtubule_images]
            if er_images:
                assert isinstance(er_images, str)
                er_images = [er_images]
            nuclei_images = [nuclei_images]

        microtubule_images = [
            os.path.expanduser(item)
            for _, item in enumerate(microtubule_images)
        ]
        nuclei_images = [
            os.path.expanduser(item) for _, item in enumerate(nuclei_images)
        ]

        microtubule_imgs = list(
            map(lambda item: imageio.imread(item), microtubule_images)
        )
        nuclei_imgs = list(
            map(lambda item: imageio.imread(item), nuclei_images)
        )
        if er_images:
            er_images = [
                os.path.expanduser(item) for _, item in enumerate(er_images)
            ]
            er_imgs = list(map(lambda item: imageio.imread(item), er_images))
        else:
            er_imgs = [
                np.zeros(item.shape, dtype=item.dtype)
                for _, item in enumerate(microtubule_imgs)
            ]
        self.cell_imgs = list(
            map(
                lambda item: np.dstack((item[0], item[1], item[2])),
                list(zip(microtubule_imgs, er_imgs, nuclei_imgs)),
            )
        )

    def label_nuclei(self, images):
        """
        Label the nuclei in all the images in the list.

        Returns either a list of labeled images or a generator which will
        yield a single labeled image at a time.

        Keyword arguments:
        images -- A list of images or a list of paths to images.
                  The images should have the nuclei in the blue channels.
        """

        def _preprocess(image):
            if isinstance(image, str):
                image = imageio.imread(image)
            self.target_shape = image.shape
            if len(image.shape) == 2:
                image = np.dstack((image, image, image))
            image = transform.rescale(
                image, self.scale_factor, multichannel=True
            )
            nuc_image = np.dstack(
                (image[..., 2], image[..., 2], image[..., 2])
            )
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

        preprocessed_images = map(_preprocess, images)
        predictions = map(lambda x: _segment_helper([x]), preprocessed_images)
        predictions = map(lambda x: x.to("cpu").numpy()[0], predictions)
        predictions = map(util.img_as_ubyte, predictions)
        predictions = list(map(self.restore_scaling_padding, predictions))
        return predictions

    def restore_scaling_padding(self, n_prediction):
        """Restore an image from scaling and padding.

           This method is intended for internal use.
           It takes the output from the nuclei model as input."""
        n_prediction = n_prediction.transpose([1, 2, 0])
        if self.padding:
            n_prediction = n_prediction[
                32 : 32 + self.scaled_shape[0],
                32 : 32 + self.scaled_shape[1],
                ...,
            ]
        if not self.scale_factor == 1:
            n_prediction[..., 0] = 0
            n_prediction = cv2.resize(
                n_prediction,
                (self.target_shape[0], self.target_shape[1]),
                interpolation=cv2.INTER_AREA,
            )
        return n_prediction

    def label_cells(self, images):
        """
        Label the cells in all the images in the list.
        Returns either a list of labeled images or a generator which will
        yield a single labeled image at a time.

        Keyword arguments:
        images -- A list of image arrays or a list of paths to image(s).
                 The images should have the nuclei channel(s) in last and
                 microtubule image(s) in the last channel, er images in the middle channel if provided. It follows the pattern like [microtubule_image, er_image/None, nuclei_image] or a list of [[microtubule_image0, microtubule_image1, ...], None/[er_image0, er_image1, ...], [nuclei_image0, nuclei_image1, ...]]
        """

        def _preprocess(image):
            if isinstance(image, str):
                image = imageio.imread(image)
                image = image / 255
            self.target_shape = image.shape
            assert len(image.shape) == 3, "image should has 3 channels"
            # cell_image = np.dstack((image, image, image))
            cell_image = transform.rescale(
                image, self.scale_factor, multichannel=True
            )
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
                imgs = torch.tensor(imgs).float()
                imgs = imgs.to(self.device)
                imgs = imgs.sub_(mean[:, None, None]).div_(std[:, None, None])

                imgs = self.cell_model(imgs)
                imgs = F.softmax(imgs, dim=1)
                return imgs

        def _postprocess(nuclei_seg, cell_seg):
            """post processing cell labels"""

            def __fill_holes(image):
                """fill_holes for labelled image, with a unique number"""
                boundaries = segmentation.find_boundaries(image)
                image = np.multiply(image, np.invert(boundaries))
                image = ndi.binary_fill_holes(image > 0)
                image = ndi.label(image)[0]
                return image

            def __wsh(
                mask_img,
                threshold,
                border_img,
                seeds,
                threshold_adjustment=0.35,
                small_object_size_cutoff=10,
            ):
                img_copy = np.copy(mask_img)
                m = seeds * border_img  # * dt
                img_copy[m <= threshold + threshold_adjustment] = 0
                img_copy[m > threshold + threshold_adjustment] = 1
                img_copy = img_copy.astype(np.bool)
                img_copy = remove_small_objects(
                    img_copy, small_object_size_cutoff
                ).astype(np.uint8)

                mask_img[mask_img <= threshold] = 0
                mask_img[mask_img > threshold] = 1
                mask_img = mask_img.astype(np.bool)
                mask_img = remove_small_holes(mask_img, 1000)
                mask_img = remove_small_objects(mask_img, 8).astype(np.uint8)
                markers = ndi.label(img_copy, output=np.uint32)[0]
                labeled_array = segmentation.watershed(
                    mask_img, markers, mask=mask_img, watershed_line=True
                )
                return labeled_array

            nuclei_label = __wsh(
                nuclei_seg[..., 2] / 255.0,
                0.4,
                1 - (nuclei_seg[..., 1] + cell_seg[..., 1]) / 255.0 > 0.05,
                nuclei_seg[..., 2] / 255,
                threshold_adjustment=-0.25,
                small_object_size_cutoff=500,
            )

            # for hpa_image, to remove the small pseduo nuclei
            nuclei_label = remove_small_objects(nuclei_label, 2500)
            nuclei_label = measure.label(nuclei_label)
            # this is to remove the cell borders' signal from cell mask.
            # could use np.logical_and with some revision, to replace this func.
            # Tuned for segmentation hpa images
            threshold_value = max(
                0.22, filters.threshold_otsu(cell_seg[..., 2] / 255) * 0.5
            )
            # exclude the green area first
            cell_region = np.multiply(
                cell_seg[..., 2] / 255 > threshold_value,
                np.invert(
                    np.asarray(cell_seg[..., 1] / 255 > 0.05, dtype=np.int8)
                ),
            )
            sk = np.asarray(cell_region, dtype=np.int8)
            distance = np.clip(
                cell_seg[..., 2], 255 * threshold_value, cell_seg[..., 2]
            )
            cell_label = segmentation.watershed(
                -distance, nuclei_label, mask=sk
            )
            cell_label = remove_small_objects(cell_label, 5500).astype(
                np.uint8
            )
            selem = disk(6)
            cell_label = closing(cell_label, selem)
            cell_label = __fill_holes(cell_label)
            # this part is to use green channel, and extend cell label to green channel
            # benefit is to exclude cells clear on border but without nucleus
            sk = np.asarray(
                np.add(
                    np.asarray(cell_label > 0, dtype=np.int8),
                    np.asarray(cell_seg[..., 1] / 255 > 0.05, dtype=np.int8),
                )
                > 0,
                dtype=np.int8,
            )
            cell_label = segmentation.watershed(-distance, cell_label, mask=sk)
            cell_label = __fill_holes(cell_label)
            cell_label = np.asarray(cell_label > 0, dtype=np.uint8)
            cell_label = measure.label(cell_label)
            cell_label = remove_small_objects(cell_label, 5500)
            cell_label = measure.label(cell_label)
            cell_label = np.asarray(cell_label, dtype=np.uint16)
            nuclei_label = np.multiply(cell_label > 0, nuclei_label) > 0
            nuclei_label = measure.label(nuclei_label)
            nuclei_label = remove_small_objects(nuclei_label, 2500)
            nuclei_label = np.multiply(cell_label, nuclei_label > 0)

            return cell_label
        
        self.batch_check(images)
        preprocessed_images = map(_preprocess, self.cell_imgs)
        predictions = map(lambda x: _segment_helper([x]), preprocessed_images)
        predictions = map(lambda x: x.to("cpu").numpy()[0], predictions)
        predictions = map(self.restore_scaling_padding, predictions)
        predictions = list(map(util.img_as_ubyte, predictions))
        if self.post_processing:
            nuclei_labels = self.label_nuclei(self.cell_imgs)
            predictions = list(
                map(
                    lambda item: _postprocess(item[0], item[1]),
                    list(zip(nuclei_labels, predictions)),
                )
            )

        return predictions
