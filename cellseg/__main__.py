import click
import imageio
from cell_segmentator import HPA_CellImage_Seg

@click.command()
@click.option('--red_channel', prompt="microtubules image path", help='The image path of microtubules')
@click.option('--blue_channel', prompt="nuclei image path", help='The image path of nuclei')
@click.option('--nuclei_model', prompt="nuclei model path", help='The model path of nuclei model')
@click.option('--cell_model', prompt="cell model path", help='The model path of cell model')
@click.option('--labeled_mask', prompt="Output file path of lableled mask", help='The output file path after cell mask prediction')



def execution(red_channel, blue_channel, nuclei_model, cell_model, labeled_mask):
    cell_mask = HPA_CellImage_Seg(red_channel, blue_channel, nuclei_model, cell_model).label_mask()
    imageio.imsave(labeled_mask, cell_mask)

if __name__=='__main__':
    execution()
    """    red_channel = '../data/hpa_dataset_v2/test/sample/1949_A2_1_red.tif'
    nuclei_channel = '../data/hpa_dataset_v2/test/sample/1949_A2_1_blue.tif'
    nuclei_model, cell_model = '../models/new_test/dpn_unet_nuclei.pth', '../models/new_test/dpn_unet_cell.pth'


    cell_mask = HPA_CellImage_Seg(red_channel, nuclei_channel, nuclei_model, cell_model).label_mask()
    imageio.imsave('../data/hpa_dataset_v2/test/sample/198888_output.png', cell_mask)"""