import click
from hpacellseg import __version__
from hpacellseg.hpacellseg import CellSegmentator
from PIL import Image


def main(
    cell_channel,
    er_channel,
    nuclei_channel,
    nuclei_model,
    cell_model,
    cell_mask,
    nuclei_mask,
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
