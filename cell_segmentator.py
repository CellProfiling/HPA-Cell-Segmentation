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


NORMALIZE = {'mean': [124 / 255, 117 / 255, 104 / 255],
             'std': [1 / (.0167 * 255)] * 3}

HIGH_THRESHOLD = 0.4
LOW_THRESHOLD = HIGH_THRESHOLD-0.25
MIN_CELL_SIZE = 130000  # TODO: Find good value for threshold


class CellSegmentator(object):
    """
    Uses pretrained DPN-Unet models to segment cells from images.
    """
    def __init__(self, nuclei_model, cell_model=None,
                 scale_factor=1.0, device='cuda'):
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

        if isinstance(nuclei_model, str):
            nuclei_model = torch.load(nuclei_model)
        if isinstance(nuclei_model, torch.nn.DataParallel) and device == 'cpu':
            nuclei_model = nuclei_model.module

        self.device = device
        self.nuclei_model = nuclei_model

        if cell_model:
            if isinstance(cell_model, str):
                cell_model = torch.load(cell_model)
            if (isinstance(cell_model, torch.nn.DataParallel) and
                    device == 'cpu'):
                cell_model = cell_model.module
            self.cell_model = cell_model
            self.cell_model = self.cell_model.to(self.device)
        self.nuclei_model = self.nuclei_model.to(self.device)
        self.scale_factor = scale_factor

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
            image = skimage.transform.rescale(image, self.scale_factor)
            nuc_image = np.dstack((image[..., 2],
                                   image[..., 2],
                                   image[..., 2]))
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
            scale_factor = n_prediction, 1/self.scale_factor
            n_prediction = skimage.transform.rescale(scale_factor)
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
            preprocessed_images = list(map(_preprocess, images))
            predictions = _segment_helper(preprocessed_images)
            predictions = map(lambda x: x.to('cpu').numpy(), predictions)
            return list(map(_postprocess, predictions))

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
            if self.scale_factor != 1.0:
                image = skimage.transform.rescale(image, self.scale_factor)
            cell_image = np.dstack((image[..., 0],
                                    np.zeros(image.shape[:-1]),
                                    image[..., 2]))
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

        def _postprocess(n_labels, c_prediction):
            c_prediction = c_prediction.transpose([1, 2, 0])
            if self.scale_factor != 1.0:
                scale_factor = c_prediction, 1/self.scale_factor
                c_prediction = skimage.transform.rescale(scale_factor)
            skele = skimage.morphology.skeletonize(
                skimage.morphology.binary_dilation(c_prediction[..., 1] > 0.1))
            skele = c_prediction[..., 2] > 0.35 - skele
            skele = np.clip(skele, 0, 1.0)
            distance = scipy.ndimage.distance_transform_edt(skele)
            cell_label = skimage.morphology.watershed(-distance, n_labels,
                                                      mask=skele)
            # Currently does NOT work! Hao will add the fill_holes function
            cell_label = fill_holes(cell_label)
            return cell_label

        nuclei_label = self.label_nuclei(images, generator)
        if generator:
            mapping = map(_preprocess, images)
            mapping = map(lambda x: _segment_helper([x]), mapping)
            mapping = map(lambda x: x.to('cpu').numpy()[0], mapping)
            mapping = itertools.starmap(lambda x, y: _postprocess(x, y),
                                        zip(nuclei_label, mapping))
            return mapping
        else:
            preprocessed_images = list(map(_preprocess, images))
            predictions = _segment_helper(preprocessed_images)
            predictions = map(lambda x: x.to('cpu').numpy(), predictions)
            return list(itertools.starmap(_postprocess,
                        zip(nuclei_label, predictions)))
