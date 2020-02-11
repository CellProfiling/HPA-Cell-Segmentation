import click
from skimage.io import imsave
from hpacellseg.hpacellseg import HPACellSeg
from hpacellseg import __version__
import sys

@click.command()
@click.option('--cell_channel', prompt="microtubules image path", help='The image path of microtubules')
@click.option('--nuclei_channel', prompt="nuclei image path", help='The image path of nuclei')
@click.option('--cell_mask', prompt="output file path of lableled mask", help='The output file path after cell mask prediction')
@click.option('--nuclei_mask', default=False, help='The output file path after nuclei mask prediction')
@click.option('--cell_model', prompt="cell model path", default="./cell_model.pth", help='The model path of cell model')
@click.option('--nuclei_model', prompt="nuclei model path", default="./nuclei_model.pth", help='The model path of nuclei model')
@click.option('--version', is_flag=True, default=False, help='Prints the version and exits')
def main(cell_channel, nuclei_channel, nuclei_model, cell_model, cell_mask, nuclei_mask, version):
    if version:
        print(__version__)
        sys.exit(0)
    cell_label, nuclei_label = HPACellSeg(cell_channel, nuclei_channel, nuclei_model, cell_model).label_mask()
    imsave(cell_mask, cell_label)
    if nuclei_mask:
        imsave(nuclei_mask, nuclei_label)

if __name__=='__main__':
    main()