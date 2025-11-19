from abc import ABC, abstractmethod

import torch
from tqdm import tqdm


class BaseEncoder(ABC):
    # Key to save the embeddings in the hdf5 embedding file
    embedding_file_key = None

    def __init__(self):
        assert (
            self.embedding_file_key is not None
        ), "You must override the embedding_file_key in the encoder class"
        assert (
            "/" not in self.embedding_file_key
        ), "The embedding_file_key cannot contain / characters"

    @abstractmethod
    def preprocess(self, imgs, actions):
        """
        Image preprocessing
        Args:
            imgs (torch.Tensor): images
            actions (torch.Tensor): actions
        Returns:
            Any: preprocessed images / actions that will be passed to encode
        """
        pass

    @abstractmethod
    def encode(self, postprocessed_imgs):
        """
        Image encoding
        Args:
            postprocessed_imgs: results from the preprocess method
        Returns:
            torch.Tensor: features
        """
        pass

    def encode_dataloader(self, dataloader, verbose=0):
        """
        Encode images from dataloader
        Args:
            dataloader (torch.utils.data.DataLoader): dataloader
            verbose (int): verbosity level
        Returns:
            torch.Tensor: features
            torch.Tensor: images
            torch.Tensor: labels
        """
        features = []
        with torch.no_grad():
            for imgs, actions, language in dataloader:
                inputs = self.preprocess(imgs, actions)
                outputs = self.encode(inputs)
                features.append(outputs.cpu())
        return torch.cat(features)


class CLIP(BaseEncoder):

    def __init__(
        self,
        model_class="openai/clip-vit-base-patch16",
        pooling=None,  # [None | "avg" | "max"]
        token_idx=None,  # [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        mm_vision_select_layer=-2,
        device="cuda",
    ):

        # init model
        from transformers import CLIPVisionModel, AutoProcessor

        self.model = CLIPVisionModel.from_pretrained(model_class)
        self.model.eval()
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_class)

        # model args
        self.embedding_file_key = "CLIP"
        self.pooling = pooling
        self.token_idx = token_idx
        self.mm_vision_select_layer = mm_vision_select_layer
        self.device = device

        super().__init__()

    def preprocess(self, imgs, actions=None):

        inputs = self.processor(images=imgs, return_tensors="pt")

        return inputs["pixel_values"].to(self.device)

    def encode(self, postprocessed_imgs):

        outputs = self.model(pixel_values=postprocessed_imgs, output_hidden_states=True)

        features = outputs.hidden_states[self.mm_vision_select_layer]

        if self.pooling is not None:
            if self.pooling == "avg":
                features = torch.mean(features, dim=1)
            elif self.pooling == "max":
                features = torch.max(features, dim=1).values

        elif self.token_idx is not None:
            features = features[:, self.token_idx]

        return features


class DINOv2(BaseEncoder):

    def __init__(
        self,
        model_class="facebook/dinov2-base",
        pooling=None,  # [None | "avg" | "max"]
        token_idx=None,  # [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        device="cuda",
    ):

        # init model
        from transformers import Dinov2Model, AutoImageProcessor

        self.model = Dinov2Model.from_pretrained(model_class)
        self.model.eval()
        self.model.to(device)
        self.processor = AutoImageProcessor.from_pretrained(model_class, use_fast=True)

        # model args
        self.embedding_file_key = "DINOv2"
        self.pooling = pooling
        self.token_idx = token_idx
        self.device = device

        super().__init__()

    def preprocess(self, imgs, actions=None):

        inputs = self.processor(images=imgs, return_tensors="pt")

        return inputs["pixel_values"].to(self.device)

    def encode(self, postprocessed_imgs):

        outputs = self.model(pixel_values=postprocessed_imgs, output_hidden_states=True)

        features = outputs.last_hidden_state

        if self.pooling is not None:
            if self.pooling == "avg":
                features = torch.mean(features, dim=1)
            elif self.pooling == "max":
                features = torch.max(features, dim=1).values

        elif self.token_idx is not None:
            # [cls] token of last layer -> self.token_idx = 0
            # https://github.com/huggingface/transformers/blob/1f9f57ab4c8c30964360a2ba697c339f6d31f03f/src/transformers/models/dinov2/modeling_dinov2.py#L711
            features = features[:, self.token_idx]

        return features


class DINOv3(BaseEncoder):

    def __init__(
        self,
        model_class="facebook/dinov3-vitb16-pretrain-lvd1689m",
        pooling=None,  # [None | "avg" | "max"]
        token_idx=None,  # [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        device="cuda",
    ):

        # init model
        from transformers import AutoModel, AutoImageProcessor

        self.model = AutoModel.from_pretrained(model_class)
        self.model.eval()
        self.model.to(device)
        self.processor = AutoImageProcessor.from_pretrained(model_class)

        # model args
        self.embedding_file_key = "DINOv3"
        self.pooling = pooling
        self.token_idx = token_idx
        self.device = device

        super().__init__()

    def preprocess(self, imgs, actions=None):

        inputs = self.processor(images=imgs, return_tensors="pt")

        return inputs["pixel_values"].to(self.device)

    def encode(self, postprocessed_imgs):

        outputs = self.model(pixel_values=postprocessed_imgs, output_hidden_states=True)

        features = outputs.last_hidden_state

        if self.pooling is not None:
            if self.pooling == "avg":
                features = torch.mean(features, dim=1)
            elif self.pooling == "max":
                features = torch.max(features, dim=1).values

        elif self.token_idx is not None:
            # [cls] token of last layer -> self.token_idx = 0
            # DINOv3 outputs: 1 class token + 4 register tokens + 196 patch tokens = 201 tokens for 224x224 images
            features = features[:, self.token_idx]

        return features
