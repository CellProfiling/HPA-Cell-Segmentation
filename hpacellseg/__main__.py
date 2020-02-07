import click
from skimage.io import imsave
from hpacellseg.hpacellseg import HPACellSeg

@click.command()
@click.option('--cell_channel', prompt="microtubules image path", help='The image path of microtubules')
@click.option('--nuclei_channel', prompt="nuclei image path", help='The image path of nuclei')
@click.option('--labeled_mask', prompt="output file path of lableled mask", help='The output file path after cell mask prediction')
@click.option('--cell_model', prompt="cell model path", default="./cell_model.pth", help='The model path of cell model')
@click.option('--nuclei_model', prompt="nuclei model path", default="./nuclei_model.pth", help='The model path of nuclei model')
def main(cell_channel, nuclei_channel, nuclei_model, cell_model, labeled_mask):
    cell_mask = HPACellSeg(cell_channel, nuclei_channel, nuclei_model, cell_model).label_mask()
    imsave(labeled_mask, cell_mask)

if __name__=='__main__':
    main()