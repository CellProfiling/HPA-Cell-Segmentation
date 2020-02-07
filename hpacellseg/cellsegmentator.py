import os
import torch
import torch.nn
import itertools
import numpy as np
import torch.nn.functional as F
import skimage.measure
import skimage.morphology
import skimage.transform
import scipy.ndimage
import imageio
import cv2
import multiprocessing
from skimage.morphology import remove_small_objects, watershed, remove_small_holes, skeletonize, binary_closing,\
    binary_dilation, binary_erosion, closing, disk
from skimage import measure, exposure, segmentation
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
import urllib.request
import click

NORMALIZE = {'mean': [124 / 255, 117 / 255, 104 / 255],
             'std': [1 / (.0167 * 255)] * 3
            }

HIGH_THRESHOLD = 0.4
LOW_THRESHOLD = HIGH_THRESHOLD-0.25
MIN_CELL_SIZE = 130000  # TODO: Find good value for threshold

class CellSegmentator(object):
    """
    Uses pretrained DPN-Unet models to segment cells from images.
    """
    def __init__(self, nuclei_model, cell_model,
                 scale_factor=1.0, device='cuda', padding=False, direct_processing=False):
        """
        Keyword arguments:
        nuclei_model -- A loaded torch nuclei segmentation model or the
                              path to a file which contains such a model.
        cell_model -- A loaded torch cell segmentation model or the
                            path to a file which contains such a model.
                            The cell segmentator argument can be None if
                            only nuclei are to be segmented. (default: None)
        scale_factor -- How much to scale images before they are fed to
                        segmentation models. Segmentations will be scaled back
                        up by 1/scale_factor to match the original image
                        (default: 1.0).
        device -- The device on which to run the models.
                  This should either be 'cpu' or 'cuda' (default: 'cuda').
        """
        if device != 'cuda' and device != 'cpu':
            raise ValueError(f'{device} is not a valid device (cuda/cpu)')
        if not device == 'cpu':
            try:
                assert torch.cuda.is_available()
            except AssertionError:
                print("No GPU found, using CPU.")
                device = 'cpu'
        self.device = device

        if isinstance(nuclei_model, str):
            nuclei_model = torch.load(nuclei_model, map_location=torch.device(self.device))
        if isinstance(nuclei_model, torch.nn.DataParallel) and device == 'cpu':
            nuclei_model = nuclei_model.module

        self.nuclei_model = nuclei_model.to(self.device)

        if isinstance(cell_model, str):
            cell_model = torch.load(cell_model, map_location=torch.device(self.device))
        #if isinstance(cell_model, torch.nn.DataParallel) and device == 'cpu':
        #    cell_model = cell_model.module
        self.cell_model = cell_model.to(self.device)

        self.scale_factor = scale_factor
        self.padding = padding
        self.direct_processing = direct_processing

    def label_nuclei(self, images, generator=False):
        """
        Label the nuclei in all the images in the list.
        Returns either a list of labeled images or a generator which will
        yield a single labeled image at a time.

        Keyword arguments:
        images -- A list of images or a list of paths to images.
                  The images should have the nuclei in the blue channels.
        generator -- If True, return a generator which yields individual
                     labeled images. Otherwise, return a list of all the
                     labeled images. (default: False)
        """
        def _preprocess(image):
            if isinstance(image, str):
                image = imageio.imread(image)
                image = image/255
            self.target_shape = image.shape
            if len(image.shape) == 2:
                image = np.dstack((image, image, image))
            image = skimage.transform.rescale(image, self.scale_factor, multichannel=True)
            nuc_image = np.dstack((image[..., 2], image[..., 2], image[..., 2]))
            if self.padding:
                rows, cols = nuc_image.shape[:2]
                self.scaled_shape = rows, cols
                nuc_image = cv2.copyMakeBorder(nuc_image, 32, (32-rows%32), 32, (32-cols%32), cv2.BORDER_REFLECT)
            nuc_image = nuc_image.transpose([2, 0, 1])
            return nuc_image

        def _segment_helper(imgs):
            with torch.no_grad():
                mean = torch.as_tensor(NORMALIZE['mean'], device=self.device)
                std = torch.as_tensor(NORMALIZE['std'], device=self.device)
                imgs = torch.tensor(imgs).float()
                imgs = imgs.to(self.device)
                imgs = imgs.sub_(mean[:, None, None]).div_(std[:, None, None])

                imgs = self.nuclei_model(imgs)
                imgs = F.softmax(imgs, dim=1)
                return imgs

        def _postprocess(n_prediction):
            n_prediction = n_prediction.transpose([1, 2, 0])
            n_prediction = skimage.transform.rescale(n_prediction, 1/self.scale_factor)
            img_copy = np.copy(n_prediction[..., 2])
            borders = (n_prediction[..., 1] > 0.05).astype(np.uint8)
            m = img_copy * (1 - borders)

            img_copy[m <= LOW_THRESHOLD] = 0
            img_copy[m > LOW_THRESHOLD] = 1
            img_copy = img_copy.astype(np.bool)
            img_copy = skimage.morphology.binary_erosion(img_copy)
            # TODO: Add parameter for remove small object size for
            #       differently scaled images.
            # img_copy = skimage.morphology.remove_small_objects(img_copy, 500)
            img_copy = img_copy.astype(np.uint8)
            markers = skimage.measure.label(img_copy).astype(np.uint32)

            mask_img = np.copy(n_prediction[..., 2])
            mask_img[mask_img <= HIGH_THRESHOLD] = 0
            mask_img[mask_img > HIGH_THRESHOLD] = 1
            mask_img = mask_img.astype(np.bool)
            mask_img = skimage.morphology.remove_small_holes(mask_img, 1000)
            # TODO: Figure out good value for remove small objects.
            # mask_img = skimage.morphology.remove_small_objects(mask_img, 8)
            mask_img = mask_img.astype(np.uint8)
            nuclei_label = skimage.morphology.watershed(mask_img, markers,
                                                        mask=mask_img,
                                                        watershed_line=True)
            return nuclei_label

        if generator:
            mapping = map(_preprocess, images)
            mapping = map(lambda x: _segment_helper([x]), mapping)
            mapping = map(lambda x: x.to('cpu').numpy()[0], mapping)
            mapping = map(_postprocess, mapping)
            return mapping
        else:
            preprocessed_images = map(_preprocess, images)
            predictions = map(lambda x: _segment_helper([x]), preprocessed_images)
            predictions = map(lambda x: x.to('cpu').numpy()[0], predictions)
            predictions = map(lambda x: img_as_ubyte(x), predictions)
            predictions = list(map(lambda x: self.restore_scaling_padding(x), predictions))
            if self.direct_processing:
                return list(map(_postprocess, predictions))
            # This is for single images
            else:
                return predictions

    def restore_scaling_padding(self, n_prediction):
            n_prediction = n_prediction.transpose([1, 2, 0])
            if self.padding:
                n_prediction = n_prediction[32: 32+self.scaled_shape[0], 32:32+self.scaled_shape[1], ... ]
            if not self.scale_factor == 1:
                n_prediction[...,0] = 0
                #n_prediction = skimage.transform.rescale(n_prediction, 1/self.scale_factor, multichannel=True)
                n_prediction = cv2.resize(n_prediction, (self.target_shape[0], self.target_shape[1]), interpolation=cv2.INTER_AREA)         
            return n_prediction


    def label_cells(self, images, generator=False):
        """
        Label the cells in all the images in the list.
        Returns either a list of labeled images or a generator which will
        yield a single labeled image at a time.

        Keyword arguments:
        images -- A list of images or a list of paths to images.
                  The images should have the nuclei in the blue channels and
                  microtubules in the red channel.
        generator -- If True, return a generator which yields individual
                     labeled images. Otherwise, return a list of all the
                     labeled images. (default: False)
        """
        def _preprocess(image):
            if isinstance(image, str):
                image = imageio.imread(image)
                image = image/255
            self.target_shape = image.shape
            assert len(image.shape) == 3, "image should has 3 channels"
                #cell_image = np.dstack((image, image, image))
            cell_image = skimage.transform.rescale(image, self.scale_factor, multichannel=True)
            if self.padding:
                rows, cols = cell_image.shape[:2]
                self.scaled_shape = rows, cols
                cell_image = cv2.copyMakeBorder(cell_image, 32, (32-rows%32), 32, (32-cols%32), cv2.BORDER_REFLECT)
            cell_image = cell_image.transpose([2, 0, 1])
            return cell_image

        def _segment_helper(imgs):
            with torch.no_grad():
                mean = torch.as_tensor(NORMALIZE['mean'], device=self.device)
                std = torch.as_tensor(NORMALIZE['std'], device=self.device)
                imgs = torch.tensor(imgs).float()
                imgs = imgs.to(self.device)
                imgs = imgs.sub_(mean[:, None, None]).div_(std[:, None, None])

                imgs = self.cell_model(imgs)
                imgs = F.softmax(imgs, dim=1)
                return imgs

        def _postprocess(nuclei_seg, cell_seg):
                """post processing cell labels"""
                def __fill_holes(image):
                    """fill_holes for labelled image, with each object has a unique number"""
                    boundaries = segmentation.find_boundaries(image)
                    image = np.multiply(image, np.invert(boundaries))
                    image = binary_fill_holes(image > 0)
                    image = ndi.label(image)[0]
                    return image

                def __wsh(mask_img, threshold, border_img, seeds, threshold_adjustment=0.35, small_object_size_cutoff=10):
                    img_copy = np.copy(mask_img)
                    m = seeds * border_img# * dt
                    img_copy[m <= threshold + threshold_adjustment] = 0
                    img_copy[m > threshold + threshold_adjustment] = 1
                    img_copy = img_copy.astype(np.bool)
                    img_copy = remove_small_objects(img_copy, small_object_size_cutoff).astype(np.uint8)

                    mask_img[mask_img <= threshold] = 0
                    mask_img[mask_img > threshold] = 1
                    mask_img = mask_img.astype(np.bool)
                    mask_img = remove_small_holes(mask_img, 1000)
                    mask_img = remove_small_objects(mask_img, 8).astype(np.uint8)
                    markers = ndi.label(img_copy, output=np.uint32)[0]
                    labeled_array = watershed(mask_img, markers, mask=mask_img, watershed_line=True)
                    return labeled_array

                nuclei_label = __wsh(nuclei_seg[...,2] / 255., \
                    0.4, 1 - (nuclei_seg[...,1]+cell_seg[..., 1]) / 255. > 0.05,nuclei_seg[...,2] / 255, threshold_adjustment=-0.25, \
                        small_object_size_cutoff=500)

                # for hpa_image, to remove the small pseduo nuclei
                # comment, found two separate nuclei regions (neighbour) with the same value. could be imporvoved.
                nuclei_label = remove_small_objects(nuclei_label, 2500).astype(np.uint8)
                # till here
                # this one is carefully set to highlight the cell border signal, iteration number. and then skeletonize to avoid trailoring the cell signals
                #sk = skeletonize(ndi.morphology.binary_dilation(cell_seg[..., 1]/255.0>0.05, iterations=2))
                # this is to remove the cell borders' signal from cell mask. could use np.logical_and with some revision, to replace this func. Tuned for segmentation hpa images
                #sk = np.subtract(np.asarray(cell_seg[...,2]/255>0.2, dtype=np.int8), np.asarray(sk, dtype=np.int8))
                # try to use threshold_otsu instead of a set value
                threshold_value = max(0.22,threshold_otsu(cell_seg[...,2]/255)*0.5)
                # exclude the green area first
                cell_region = np.multiply(cell_seg[...,2]/255>threshold_value, np.invert(np.asarray(cell_seg[...,1]/255>0.05, dtype=np.int8)))
                #cell_region = np.add(np.asarray(cell_seg[...,2]/255>threshold_value, dtype=np.int8), np.asarray(cell_seg[...,1]/255>0.05, dtype=np.int8)) > 0
                #sk = np.subtract(np.asarray(cell_seg[...,2]/255>threshold_value, dtype=np.int8), np.asarray(sk, dtype=np.int8), dtype=np.int8)
                sk = np.asarray(cell_region, dtype=np.int8)
                #sk = np.clip(sk, 0, 1.0)
                # discard distance map
                ##distance = ndi.distance_transform_edt(sk)
                ##cell_label = watershed(-distance, nuclei_label, mask=sk)
                # use cell blue channel as distance map directly
                distance = np.clip(cell_seg[...,2], 255*threshold_value, cell_seg[...,2])
                cell_label = watershed(-distance, nuclei_label, mask=sk)
                cell_label = remove_small_objects(cell_label, 5500).astype(np.uint8)
                selem = disk(6)
                cell_label = closing(cell_label, selem)
                cell_label = __fill_holes(cell_label)
                # this part is to use green channel, and extend cell label to green channel
                # benefit is to exclude cells clear on border but without nucleus
                sk = np.asarray(np.add(np.asarray(cell_label>0, dtype=np.int8), np.asarray(cell_seg[...,1]/255>0.05, dtype=np.int8)) > 0, dtype=np.int8)
                cell_label = watershed(-distance, cell_label, mask=sk)
                cell_label = __fill_holes(cell_label)
                cell_label = np.asarray(cell_label > 0, dtype=np.uint8)
                cell_label = skimage.measure.label(cell_label)
                cell_label = remove_small_objects(cell_label, 5500)               
                cell_label = skimage.measure.label(cell_label)
                cell_label = np.asarray(cell_label, dtype=np.uint16)              
                
                return cell_label
     
        if generator:
            nuclei_label = self.label_nuclei(images, generator)
            mapping = map(_preprocess, images)
            mapping = map(lambda x: _segment_helper([x]), mapping)
            mapping = map(lambda x: x.to('cpu').numpy()[0], mapping)
            mapping = itertools.starmap(lambda x,y: _postprocess(x, y),
                                        zip(nuclei_label, mapping))
            return mapping
        if self.direct_processing:
            nuclei_label = self.label_nuclei(images, generator)
            preprocessed_images = list(map(_preprocess, images))
            predictions = _segment_helper(preprocessed_images)
            predictions = map(lambda x: x.to('cpu').numpy()[0], predictions)
            return list(itertools.starmap(_postprocess, 
                        zip(nuclei_label, predictions)))
        else:
            nuclei_labels = self.label_nuclei(images, generator)
            preprocessed_images = map(_preprocess, images)
            predictions = map(lambda x: _segment_helper([x]), preprocessed_images)
            predictions = map(lambda x: x.to('cpu').numpy()[0], predictions)
            predictions = list(map(lambda x: self.restore_scaling_padding(x), predictions))
            predictions = map(lambda x: img_as_ubyte(x), predictions)
            cell_masks = list(map(lambda item: _postprocess(item[0], item[1]), list(zip(nuclei_labels, predictions))))

            return cell_masks
