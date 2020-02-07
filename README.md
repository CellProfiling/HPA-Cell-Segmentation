# HPA-Cell-Segmentation

A class for cell segmentation using pretrained U-Nets.

# Steps
- cd HPA-Cell-Segmentation
- pip3 install .

# run example

- python3 -m hpacellseg --cell_channel 'path/to/1949_A2_1_red.tif' --nuclei_channel 'path/to/1949_A2_1_blue.tif' --cell_model 'path/to/models/dpn_unet_cell.pth' --nuclei_model 'path/to/models/dpn_unet_nuclei.pth' --cell_mask 'path/to/output/cell_mask.tif' --nuclei_mask 'path/to/output/nuclei_mask.tif'
    - cell_channel : image file path of microtubules image, like folder_path_to/1948_A1_2_red.tif.
    - nuclei_channel: image file path of nuclei image, like folder_path_to/1948_A1_2_blue.tif.
    - nuclei_model: model file path of nuclei model, with specifying model file name, like folder_path_to/dpn_unet_nuclei.pth; Will automatically download the model if the model file path is invalid.
    - cell_model: model file path of cell model, with specifying model file name, like folder_path_to/dpn_unet_cell.pth; Will automatically download the model if the model file path is invalid.
    - cell_mask: cell mask file path, with specifying the file name, like folder_path_to/1948_A1_2_cell_mask.tif
    - nuclei_mask: nuclei mask file path, with specifying the file name, like folder_path_to/1948_A1_2_nuclei_mask.tif
- use .tif image for output for cell mask, like mask.tif. .tif image supports 16bit depth image
