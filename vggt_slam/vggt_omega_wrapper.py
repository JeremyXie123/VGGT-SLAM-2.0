import torch
import torch.nn as nn
from vggt_omega.models.vggt_omega import VGGTOmega

# TODO: Determine if VGGT-Omega has a layer that we can use for image verification (loop closure), similar to VGGT-SPARK.

class VGGTOmegaModel(nn.Module):
    """Adapts VGGT-Omega to the model(images, query_points, compute_similarity) interface VGGT-SPARK provides."""

    def __init__(self, checkpoint_path):
        super().__init__()
        self.model = VGGTOmega()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    def forward(self, images, query_points=None, compute_similarity=False):
        predictions = self.model(images)
        if compute_similarity:
            predictions["image_match_ratio"] = torch.ones((), device=images.device, dtype=torch.float32)
        return predictions
