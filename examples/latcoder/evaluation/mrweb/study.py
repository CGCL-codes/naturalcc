from .emd_similarity import emd_similarity, process_imgs
import os
from tqdm import tqdm
import pandas as pd
import time
import random
import sqlite3
import open_clip
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import lpips
import torch
from PIL import Image
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd


class CLIPScorer:
    def __init__(self, model_name='ViT-B-32-quickgelu', pretrained='openai'):
        """
        Initializes the CLIPScorer with the specified model.

        Args:
            model_name (str): The name of the CLIP model to use.
            pretrained (str): Specifies whether to load pre-trained weights.
        """
        self.device = "cuda" if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(self.device)

    def score(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculates the CLIP score (cosine similarity) between two images.

        Args:
            img1 (Image.Image): The first image as a PIL Image.
            img2 (Image.Image): The second image as a PIL Image.

        Returns:
            float: The cosine similarity score between the two images.
        """
        # Preprocess the images
        image1 = self.preprocess(img1).unsqueeze(0).to(self.device)
        image2 = self.preprocess(img2).unsqueeze(0).to(self.device)

        # Get the image features from CLIP using openclip
        with torch.no_grad():
            image1_features = self.model.encode_image(image1)
            image2_features = self.model.encode_image(image2)

        # Normalize the features to unit length
        image1_features /= image1_features.norm(dim=-1, keepdim=True)
        image2_features /= image2_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity between the two image features
        cosine_similarity = torch.nn.functional.cosine_similarity(image1_features, image2_features)
        return cosine_similarity.item()


class LPIPSScorer:
    def __init__(self, net='vgg'):
        """
        Initializes the LPIPS scorer with the specified network type.

        Args:
            net (str): The network to use for LPIPS calculation ('vgg', 'alex', or 'squeeze').
        """
        self.loss_fn = lpips.LPIPS(net=net)  # Load the LPIPS model
        self.device = "cuda" if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        self.loss_fn.to(self.device)

    def score(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Calculates the LPIPS similarity score between two images.

        Args:
            image1 (Image.Image): The first image as a PIL Image.
            image2 (Image.Image): The second image as a PIL Image.

        Returns:
            float: The LPIPS similarity score between the two images.
        """
        image1, image2 = process_imgs(image1, image2, 512)
        # 
        # Convert images to float tensors
        transform = transforms.Compose([
            transforms.ToTensor(),          # Convert to tensor
            transforms.Lambda(lambda x: x.to(dtype=torch.float32)),  # Convert to float32
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        img1_tensor = transform(image1).unsqueeze(0).to(self.device)
        img2_tensor = transform(image2).unsqueeze(0).to(self.device)

        # Calculate the LPIPS similarity score
        with torch.no_grad():  # Disable gradients for inference
            lpips_score = self.loss_fn(img1_tensor, img2_tensor)

        return lpips_score.item()


def ssim_score(img1, img2):
    # resize images to match the size of the smaller image
    img1, img2 = process_imgs(img1, img2, 512)
    return ssim(img1, img2, channel_axis=-1, data_range=255)

def psnr_score(img1, img2):
    """peak signal-to-noise ratio, it is a pixel-based metric"""
    img1, img2 = process_imgs(img1, img2, 512)
    return psnr(img1, img2)

def mae_score(img1, img2):
    """mean absolute error, it is a pixel-based metric"""
    img1, img2 = process_imgs(img1, img2, 512)
    # max_mae = np.mean(np.maximum(img1, 255 - img1))
    mae = np.mean(np.abs(img1 - img2))
    # return {"mae": mae, "normalized_mae": 1 - mae / max_mae}
    return mae
