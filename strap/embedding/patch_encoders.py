"""
Encoders that save unpooled patch-level features for spatial visualization.

These encoders DO NOT pool the patch features, preserving spatial structure.
"""

from strap.embedding.encoders import BaseEncoder
import torch


class DINOv3_Patches(BaseEncoder):
    """
    DINOv3 encoder that saves unpooled patch features.

    Output shape: (num_patches, embed_dim) instead of (embed_dim,)
    For 224x224 images with 16x16 patches: (196, 768)
    """

    def __init__(
        self,
        model_class="facebook/dinov3-vitb16-pretrain-lvd1689m",
        device="cuda",
    ):
        # init model
        from transformers import AutoModel, AutoImageProcessor

        self.model = AutoModel.from_pretrained(model_class)
        self.model.eval()
        self.model.to(device)
        self.processor = AutoImageProcessor.from_pretrained(model_class)

        # model args
        self.embedding_file_key = "DINOv3_patches"
        self.device = device

        super().__init__()

    def preprocess(self, imgs, actions=None):
        inputs = self.processor(images=imgs, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def encode(self, postprocessed_imgs):
        outputs = self.model(pixel_values=postprocessed_imgs, output_hidden_states=True)
        features = outputs.last_hidden_state  # (batch, num_tokens, embed_dim)

        # DINOv3 tokens: 1 CLS + 4 register + 196 patches (for 224x224)
        # Skip CLS and register tokens, keep only patches
        patch_features = features[:, 5:]  # (batch, 196, 768)

        return patch_features


class DINOv2_Patches(BaseEncoder):
    """
    DINOv2 encoder that saves unpooled patch features.

    Output shape: (num_patches, embed_dim) instead of (embed_dim,)
    For 224x224 images with 16x16 patches: (196, 768)
    """

    def __init__(
        self,
        model_class="facebook/dinov2-base",
        device="cuda",
    ):
        # init model
        from transformers import Dinov2Model, AutoImageProcessor

        self.model = Dinov2Model.from_pretrained(model_class)
        self.model.eval()
        self.model.to(device)
        self.processor = AutoImageProcessor.from_pretrained(model_class, use_fast=True)

        # model args
        self.embedding_file_key = "DINOv2_patches"
        self.device = device

        super().__init__()

    def preprocess(self, imgs, actions=None):
        inputs = self.processor(images=imgs, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def encode(self, postprocessed_imgs):
        outputs = self.model(pixel_values=postprocessed_imgs, output_hidden_states=True)
        features = outputs.last_hidden_state  # (batch, num_tokens, embed_dim)

        # DINOv2 tokens: 1 CLS + 196 patches (for 224x224)
        # Skip CLS token, keep only patches
        patch_features = features[:, 1:]  # (batch, 196, 768)

        return patch_features
