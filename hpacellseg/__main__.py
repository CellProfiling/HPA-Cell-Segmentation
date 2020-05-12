import click
from hpacellseg import __version__
from hpacellseg.hpacellseg import HPACellSeg
from PIL import Image


@click.command()
@click.option(
    "--cell_channel",
    prompt="microtubules image path",
    help="The image path of microtubules",
)
@click.option(
    "--er_channel",
    default=None,
    help="The image path of ER",
)
@click.option(
    "--nuclei_channel",
    prompt="nuclei image path",
    help="The image path of nuclei"
)
@click.option(
    "--cell_mask",
    prompt="output file path of lableled mask",
    help="The output file path after cell mask prediction",
)
@click.option(
    "--nuclei_mask",
    default=False,
    help="The output file path after nuclei mask prediction",
)
@click.option(
    "--cell_model",
    prompt="cell model path",
    default="./cell_model.pth",
    help="The model path of cell model",
)
@click.option(
    "--nuclei_model",
    prompt="nuclei model path",
    default="./nuclei_model.pth",
    help="The model path of nuclei model",
)
@click.version_option(__version__, message="%(version)s")
def main(
    cell_channel,
    er_channel,
    nuclei_channel,
    nuclei_model,
    cell_model,
    cell_mask,
    nuclei_mask
):
    image_channels = [cell_channel, er_channel, nuclei_channel]
    if er_channel:
        # rename the cell model path, to avoid using the save model that does
        # not support 3 channel segmentation prediction
        cell_model = cell_model[:-4] + "_3ch.pth"
    cell_label, nuclei_label = HPACellSeg(
        image_channels, nuclei_model, cell_model
    ).label_mask(scale_factor=0.25)
    Image.fromarray(cell_label).save(cell_mask, bits=16)
    if nuclei_mask:
        Image.fromarray(nuclei_label).save(nuclei_mask, bits=16)


if __name__ == "__main__":
    main()
