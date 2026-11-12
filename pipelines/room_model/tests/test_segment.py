import numpy as np
import pytest

from room_model import anchor, ortho
from room_model.ortho import FaceFrame
from synthetic import FacePatch, render_face

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="SAM tests need CUDA")

FRAME = FaceFrame(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))


@pytest.fixture(scope="module")
def segmenter():
    from room_model.segment import Segmenter

    return Segmenter()


def test_sam_prompted_by_the_scan_rectangle_finds_the_leaf(segmenter):
    # a leaf hung a few millimetres inside its opening, the scan rectangle a few centimetres too large
    patches = [FacePatch((0.504, 1.296), (0.005, 2.096), -0.04, 110),
               FacePatch((-2, 0.50), (-1, 4), 0.0, 190), FacePatch((1.30, 3), (-1, 4), 0.0, 190),
               FacePatch((0.50, 1.30), (2.10, 4), 0.0, 190), FacePatch((-2, 3), (-1, 4), -0.6, 40)]
    image, photo = render_face(FRAME, patches, (0.9, 1.3, 2.2), (0.9, 1.05, 0.0))
    o = ortho.ortho(photo, image, FRAME, (0.30, 1.50, -0.15, 2.30))
    mask = anchor.leaf_mask(o, (0.47, 1.34, 0.0, 2.13), segmenter)
    truth = ((o.ss > 0.504) & (o.ss < 1.296))[None, :] & ((o.ys > 0.005) & (o.ys < 2.096))[:, None]
    assert mask is not None and (mask & truth).sum() / (mask | truth).sum() > 0.9
    assert segmenter.diagnostics()["device"] == "cuda"
