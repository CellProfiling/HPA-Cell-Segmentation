import click
from hpacellseg import __version__
from hpacellseg.cellsegmentator import CellSegmentator
from hpacellseg.utils import label_nuclei, label_cell


def main(images=None):
    """This part is for implementation, you can customize for your own usage."""
    segmentator = CellSegmentator(
        nuclei_model="./nuclei_model.pth",
        cell_model="./cell_3ch_model.pth",
        scale_factor=0.25,
        device="cuda:1",
        padding=True,
        multi_channel_model=False,
    )

    nuclei_preds = segmentator.pred_nuclei(images[2])
    cell_preds = segmentator.pred_cells(images)

    nuclei_pred = nuclei_preds[0]
    cell_preds = cell_preds[0]
    # this is the post-processing part
    # this will give you both cell_mask and nuclei _mask
    nuclei_mask, cell_mask = label_cell(nuclei_pred, cell_pred)
    # this is for nuclei mask generation
    nuclei_mask = label_nuclei(nuclei_pred)

    # get what ever you want to get
    return nuclei_preds, cell_preds

if __name__ == "__main__":
    main()
