# HPA-Cell-image-segmentation

A class for cell segmentation using pretrained U-Nets.

# Steps
- cd HPA-Cell-image-segmentation
- git checkout hpa-image-test
- sh install.sh

# run example

- python -m cellseg --red_channel 'Path/to/1949_A2_1_red.tif' --blue_channel 'Path/to/1949_A2_1_blue.tif' --nuclei_model 'Path/to/models/dpn_unet_nuclei.pth' --cell_model 'Path/to/models/dpn_unet_cell.pth' --labeled_mask 'Path/to/output/mask.png'