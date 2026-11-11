"""SAM 2.1 (Hugging Face transformers port) prompted with a box, and optionally points, on an RGB image."""
from __future__ import annotations

import numpy as np

MODEL_ID = "facebook/sam2.1-hiera-large"
REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"  # pinned snapshot: masks must not change under a cached run


class Segmenter:
    """Callable `(rgb, box, points=None) -> [(mask, predicted IoU)]`, the three SAM masks of one prompt.

    `box` is [x0, y0, x1, y1] in pixels, `points` a list of [x, y] taken as foreground. The model loads on the
    first call, on the GPU when CUDA is available, else on the CPU (much slower).
    """

    def __init__(self, device: str | None = None):
        self._device, self._model, self._processor = device, None, None

    def _load(self):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = Sam2Processor.from_pretrained(MODEL_ID, revision=REVISION)
        self._model = Sam2Model.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.float32).to(self._device).eval()

    def __call__(self, rgb: np.ndarray, box, points=None) -> list[tuple[np.ndarray, float]]:
        import torch

        if self._model is None:
            self._load()
        kw = {"input_boxes": [[[float(v) for v in box]]]}
        if points:
            kw.update(input_points=[[[[float(x), float(y)] for x, y in points]]], input_labels=[[[1] * len(points)]])
        inputs = self._processor(images=rgb, return_tensors="pt", **kw).to(self._device)
        with torch.no_grad():
            out = self._model(**inputs, multimask_output=True)
        masks = self._processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0][0].numpy() > 0
        scores = out.iou_scores[0, 0].float().cpu().numpy()
        return [(masks[k], float(scores[k])) for k in range(len(scores))]

    def diagnostics(self) -> dict:
        import torch

        info = {"model": MODEL_ID, "device": self._device}
        if self._device == "cuda":
            info["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        return info
