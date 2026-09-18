import os
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image


class SemanticMasker:
    """Segments the given object types with SAM 3, as a prior on what is likely to move."""

    def __init__(self, prompts=("person",), confidence_threshold=0.5):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam3Processor(
            build_sam3_image_model(device=self.device), confidence_threshold=confidence_threshold
        )
        self.prompts = tuple(prompts)

    def detect(self, frame):
        """Returns {prompt: (N, H, W) bool}, one mask per detected instance of that prompt."""
        # SAM 3's fused MLP hardcodes bfloat16, so without autocast it raises a dtype mismatch.
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
            # One encode per frame; set_text_prompt overwrites just the text features.
            state = self.processor.set_image(to_pil_image(frame))
            found = {}
            for prompt in self.prompts:
                masks = self.processor.set_text_prompt(state=state, prompt=prompt)["masks"]
                # (N, 1, H, W), one mask per detection, all of them kept.
                found[prompt] = masks.cpu().numpy().reshape(-1, *frame.shape[-2:])
        return found

    def compute(self, pred_dict):
        """Returns one {prompt: (N, H, W) bool} per frame, instances kept apart."""
        frames = torch.as_tensor(np.asarray(pred_dict["images"]))  # (S, 3, H, W) in [0, 1]
        return [self.detect(frame) for frame in frames]

    @staticmethod
    def flatten(per_frame):
        """Collapses the instances of each frame into {prompt: (S, H, W) bool}."""
        prompts = per_frame[0].keys() if per_frame else []
        return {p: np.stack([f[p].any(axis=0) for f in per_frame]) for p in prompts}

    def save(self, pred_dict, image_names, out_root):
        """Writes the union mask and its overlays, and reports coverage."""
        per_frame = self.compute(pred_dict)
        by_prompt = self.flatten(per_frame)
        dynamic = self.union(by_prompt)
        self.write_masks(dynamic, image_names, os.path.join(out_root, "masks_semantic"))
        self.write_overlays(dynamic, pred_dict, image_names, os.path.join(out_root, "overlays_semantic"))
        for prompt, mask in by_prompt.items():
            per_pixel = mask.mean(axis=(1, 2))
            counts = [len(f[prompt]) for f in per_frame]
            print(f"Semantic mask '{prompt}' covered {mask.mean():.2%} of pixels (per frame {per_pixel.min():.2%} to {per_pixel.max():.2%}, {min(counts)} to {max(counts)} instances)")
        if len(by_prompt) > 1:
            print(f"Semantic mask covered {dynamic.mean():.2%} of pixels in total")
        return dynamic

    @staticmethod
    def union(masks_by_prompt):
        """Merges the per-prompt masks into the one mask reconstruction consumes."""
        masks = list(masks_by_prompt.values())
        return np.logical_or.reduce(masks) if masks else None

    def write_masks(self, masks, image_names, out_dir):
        """One PNG per frame, 255 where dynamic, at the model's resolution and not the source's."""
        os.makedirs(out_dir, exist_ok=True)
        for mask, image_name in zip(masks, image_names):
            stem = os.path.splitext(os.path.basename(image_name))[0]
            cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), mask.astype(np.uint8) * 255)

    def write_overlays(self, masks, pred_dict, image_names, out_dir):
        """Same masks drawn over their frames, for looking at rather than for downstream use."""
        os.makedirs(out_dir, exist_ok=True)
        # Frames from pred_dict, not reloaded, so they line up with the masks exactly.
        frames = (np.asarray(pred_dict["images"]).transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        for mask, frame, image_name in zip(masks, frames, image_names):
            panel = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            panel[mask] = (0.35 * panel[mask] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel, contours, -1, (0, 255, 255), 1)
            stem = os.path.splitext(os.path.basename(image_name))[0]
            cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), panel)
