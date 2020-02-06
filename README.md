# HPA-Cell-image-segmentation

A class for cell segmentation using pretrained U-Nets.

# Steps
- cd HPA-Cell-image-segmentation
- git checkout hpa-image-seg
- sh install.sh

# run example

- python -m cellseg --red_channel 'Path/to/1949_A2_1_red.tif' --blue_channel 'Path/to/1949_A2_1_blue.tif' --nuclei_model 'Path/to/models/dpn_unet_nuclei.pth' --cell_model 'Path/to/models/dpn_unet_cell.pth' --labeled_mask 'Path/to/output/mask.tif'
    - red_channel : image file path of microtubules image, like folder_path_to/1948_A1_2_red.tif.
    - blue_channel: image file path of nuclei image, like folder_path_to/1948_A1_2_blue.tif.
    - nuclei_model: model file path of nuclei model, with specifying model file name, like folder_path_to/dpn_unet_nuclei.pth; Will automatically download the model if the model file path is invalid.
    - cell_model: model file path of cell model, with specifying model file name, like folder_path_to/dpn_unet_cell.pth; Will automatically download the model if the model file path is invalid.
    - labeled_mask: cell mask file path, with specifying the file name, like folder_path_to/1948_A1_2_mask.tif
- use .tif image for output for cell mask, like mask.tif. .tif image supports 16bit depth image