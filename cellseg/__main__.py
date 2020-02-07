import click
from skimage.io import imsave
from cellseg.cell_segmentator import HPA_CellImage_Seg

@click.command()
@click.option('--red_channel', prompt="microtubules image path", help='The image path of microtubules')
@click.option('--blue_channel', prompt="nuclei image path", help='The image path of nuclei')
@click.option('--nuclei_model', prompt="nuclei model path", help='The model path of nuclei model')
@click.option('--cell_model', prompt="cell model path", help='The model path of cell model')
@click.option('--labeled_mask', prompt="Output file path of lableled mask", help='The output file path after cell mask prediction')
@click.option('--nuclei_mask', 'nuclei_file', required=False, prompt="Output file path of nuclei mask",
              help='The output file path after nuclei mask prediction')
def execution(red_channel, blue_channel, nuclei_model, cell_model, labeled_mask, nuclei_file):
    cell_mask, nuclei_mask = HPA_CellImage_Seg(red_channel, blue_channel, nuclei_model, cell_model).label_mask()
    imsave(labeled_mask, cell_mask)
    if nuclei_file:
        imsave(nuclei_file, nuclei_mask)


if __name__=='__main__':
    execution()