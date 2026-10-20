#!/usr/bin/env python3
"""
Image Quality Scorer using LAION CLIP model
Scores images on a 1-10 scale based on quality indicators
"""

import logging

import cv2
import numpy as np
import torch
from PIL import Image

from .model_cache import get_model_cache
from .config import (
    AESTHETIC_FALLBACK_MODEL,
    AESTHETIC_FALLBACK_PRETRAINED,
    clip_model_from_env,
    clip_pretrained_from_env,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class CLIPQualityScorer:
    # 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K' slow MAE: 1.069 Corr: 0.864
    # 'hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K' MAE: 2.275, Corr: 0.344
    # 'hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K' MAE: 2.600, Corr: 0.132 (worst)
    # 'hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K' MAE: 1.372, Corr: 0.747 (BEST BALANCE)
    # 'ViT-B-32' base (same as above with laion2b weights)
    def __init__(self, model_name=None, pretrained=None):
        # Use environment variables if not provided
        if model_name is None:
            env_model = clip_model_from_env()
            env_pretrained = clip_pretrained_from_env()
            if env_model:
                model_name = env_model
                if env_pretrained and pretrained is None:
                    pretrained = env_pretrained
            else:
                # Fallback to default
                # model_name = 'hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K'
                model_name = AESTHETIC_FALLBACK_MODEL
                pretrained = AESTHETIC_FALLBACK_PRETRAINED
                logger.info(f"Fallback used: [{model_name}, {pretrained}]")

        self._model_name = model_name
        self._pretrained = pretrained
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy loading - model will be loaded on first use
        self._model = None
        self._preprocess = None
        self._tokenizer = None

        # Multi-dimensional quality assessment

        # Dimension 1: Product clarity (most important)
        self.clarity_positive = [
            "sharp in-focus product",
            "clear detailed product image",
            "well-lit product photo",
        ]
        self.clarity_negative = [
            "blurry out-of-focus product",
            "unclear product image",
            "dark poorly lit photo",
        ]

        # Dimension 2: Composition (important)
        self.composition_positive = [
            "well-framed product photo",
            "good product composition",
            "professional product angle",
        ]
        self.composition_negative = [
            "poorly framed product",
            "bad product composition",
            "awkward product angle",
        ]

        # Dimension 3: Watermarks/Text (critical defect)
        self.defect_positive = [
            "clean product photo without text",
            "product photo without watermark",
            "unobstructed product view",
        ]
        self.defect_negative = [
            "watermark on product photo",
            "text overlay on product",
            "obstructed product view",
        ]

    def _load_model(self):
        """Lazy load the CLIP model using the shared cache."""
        if self._model is None:
            logger.info(f"Loading CLIP model: {self._model_name} (pretrained={self._pretrained})")
            cache = get_model_cache()
            self._model, self._preprocess, self._tokenizer, _ = cache.get_clip_model(
                self._model_name,
                pretrained=self._pretrained,
                device=self.device,
            )

    @property
    def model(self):
        """Lazy-loaded model."""
        self._load_model()
        return self._model

    @property
    def preprocess(self):
        """Lazy-loaded preprocess function."""
        self._load_model()
        return self._preprocess

    @property
    def tokenizer(self):
        """Lazy-loaded tokenizer."""
        self._load_model()
        return self._tokenizer

    def score_image(self, image_path):
        """
        Score an image from 1-10 based on quality
        Returns: score (float), positive_score, negative_score
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            return 0.0, 0.0, 0.0

        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get image features
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Dimension 1: Clarity
            clarity_pos_tokens = self.tokenizer(self.clarity_positive).to(self.device)
            clarity_pos_features = self.model.encode_text(clarity_pos_tokens)
            clarity_pos_features = clarity_pos_features / clarity_pos_features.norm(dim=-1, keepdim=True)
            clarity_pos_sim = (image_features @ clarity_pos_features.T).mean().item()

            clarity_neg_tokens = self.tokenizer(self.clarity_negative).to(self.device)
            clarity_neg_features = self.model.encode_text(clarity_neg_tokens)
            clarity_neg_features = clarity_neg_features / clarity_neg_features.norm(dim=-1, keepdim=True)
            clarity_neg_sim = (image_features @ clarity_neg_features.T).mean().item()

            clarity_score = clarity_pos_sim - clarity_neg_sim

            # Dimension 2: Composition
            comp_pos_tokens = self.tokenizer(self.composition_positive).to(self.device)
            comp_pos_features = self.model.encode_text(comp_pos_tokens)
            comp_pos_features = comp_pos_features / comp_pos_features.norm(dim=-1, keepdim=True)
            comp_pos_sim = (image_features @ comp_pos_features.T).mean().item()

            comp_neg_tokens = self.tokenizer(self.composition_negative).to(self.device)
            comp_neg_features = self.model.encode_text(comp_neg_tokens)
            comp_neg_features = comp_neg_features / comp_neg_features.norm(dim=-1, keepdim=True)
            comp_neg_sim = (image_features @ comp_neg_features.T).mean().item()

            composition_score = comp_pos_sim - comp_neg_sim

            # Dimension 3: Defects (watermarks, etc.)
            defect_pos_tokens = self.tokenizer(self.defect_positive).to(self.device)
            defect_pos_features = self.model.encode_text(defect_pos_tokens)
            defect_pos_features = defect_pos_features / defect_pos_features.norm(dim=-1, keepdim=True)
            defect_pos_sim = (image_features @ defect_pos_features.T).mean().item()

            defect_neg_tokens = self.tokenizer(self.defect_negative).to(self.device)
            defect_neg_features = self.model.encode_text(defect_neg_tokens)
            defect_neg_features = defect_neg_features / defect_neg_features.norm(dim=-1, keepdim=True)
            defect_neg_sim = (image_features @ defect_neg_features.T).mean().item()

            defect_score = defect_pos_sim - defect_neg_sim

        # Weighted combination (optimized from fine-tuning)
        raw_score = (
            -0.038554 * clarity_score +
            0.396935 * composition_score +
            0.641619 * defect_score
        )

        # Linear scaling from fine-tuning
        final_score = 171.740104 * raw_score + 6.025378
        final_score = np.clip(final_score, 1.0, 10.0)

        # Return combined score
        return final_score, raw_score, clarity_score

    def score(self, img):
        """
        Score an image from 1-10 based on quality (PIL Image or numpy array)
        Returns: float score (1-10)
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Ensure RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')

        image_input = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get image features
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Dimension 1: Clarity
            clarity_pos_tokens = self.tokenizer(self.clarity_positive).to(self.device)
            clarity_pos_features = self.model.encode_text(clarity_pos_tokens)
            clarity_pos_features = clarity_pos_features / clarity_pos_features.norm(dim=-1, keepdim=True)
            clarity_pos_sim = (image_features @ clarity_pos_features.T).mean().item()

            clarity_neg_tokens = self.tokenizer(self.clarity_negative).to(self.device)
            clarity_neg_features = self.model.encode_text(clarity_neg_tokens)
            clarity_neg_features = clarity_neg_features / clarity_neg_features.norm(dim=-1, keepdim=True)
            clarity_neg_sim = (image_features @ clarity_neg_features.T).mean().item()

            clarity_score = clarity_pos_sim - clarity_neg_sim

            # Dimension 2: Composition
            comp_pos_tokens = self.tokenizer(self.composition_positive).to(self.device)
            comp_pos_features = self.model.encode_text(comp_pos_tokens)
            comp_pos_features = comp_pos_features / comp_pos_features.norm(dim=-1, keepdim=True)
            comp_pos_sim = (image_features @ comp_pos_features.T).mean().item()

            comp_neg_tokens = self.tokenizer(self.composition_negative).to(self.device)
            comp_neg_features = self.model.encode_text(comp_neg_tokens)
            comp_neg_features = comp_neg_features / comp_neg_features.norm(dim=-1, keepdim=True)
            comp_neg_sim = (image_features @ comp_neg_features.T).mean().item()

            composition_score = comp_pos_sim - comp_neg_sim

            # Dimension 3: Defects (watermarks, etc.)
            defect_pos_tokens = self.tokenizer(self.defect_positive).to(self.device)
            defect_pos_features = self.model.encode_text(defect_pos_tokens)
            defect_pos_features = defect_pos_features / defect_pos_features.norm(dim=-1, keepdim=True)
            defect_pos_sim = (image_features @ defect_pos_features.T).mean().item()

            defect_neg_tokens = self.tokenizer(self.defect_negative).to(self.device)
            defect_neg_features = self.model.encode_text(defect_neg_tokens)
            defect_neg_features = defect_neg_features / defect_neg_features.norm(dim=-1, keepdim=True)
            defect_neg_sim = (image_features @ defect_neg_features.T).mean().item()

            defect_score = defect_pos_sim - defect_neg_sim

        # Weighted combination (optimized from fine-tuning)
        raw_score = (
            -0.038554 * clarity_score +
            0.396935 * composition_score +
            0.641619 * defect_score
        )

        # Linear scaling from fine-tuning
        final_score = 171.740104 * raw_score + 6.025378
        final_score = np.clip(final_score, 1.0, 10.0)

        return float(final_score)

    def score_detailed(self, img):
        """
        Score an image with detailed breakdown of all aesthetic components.

        Args:
            img: PIL Image or numpy array

        Returns:
            dict with clarity_score, composition_score, defect_score, aesthetic_overall
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        image_input = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Dimension 1: Clarity
            clarity_pos_tokens = self.tokenizer(self.clarity_positive).to(self.device)
            clarity_pos_features = self.model.encode_text(clarity_pos_tokens)
            clarity_pos_features = clarity_pos_features / clarity_pos_features.norm(dim=-1, keepdim=True)
            clarity_pos_sim = (image_features @ clarity_pos_features.T).mean().item()

            clarity_neg_tokens = self.tokenizer(self.clarity_negative).to(self.device)
            clarity_neg_features = self.model.encode_text(clarity_neg_tokens)
            clarity_neg_features = clarity_neg_features / clarity_neg_features.norm(dim=-1, keepdim=True)
            clarity_neg_sim = (image_features @ clarity_neg_features.T).mean().item()

            clarity_score = clarity_pos_sim - clarity_neg_sim

            # Dimension 2: Composition
            comp_pos_tokens = self.tokenizer(self.composition_positive).to(self.device)
            comp_pos_features = self.model.encode_text(comp_pos_tokens)
            comp_pos_features = comp_pos_features / comp_pos_features.norm(dim=-1, keepdim=True)
            comp_pos_sim = (image_features @ comp_pos_features.T).mean().item()

            comp_neg_tokens = self.tokenizer(self.composition_negative).to(self.device)
            comp_neg_features = self.model.encode_text(comp_neg_tokens)
            comp_neg_features = comp_neg_features / comp_neg_features.norm(dim=-1, keepdim=True)
            comp_neg_sim = (image_features @ comp_neg_features.T).mean().item()

            composition_score = comp_pos_sim - comp_neg_sim

            # Dimension 3: Defects (watermarks, etc.)
            defect_pos_tokens = self.tokenizer(self.defect_positive).to(self.device)
            defect_pos_features = self.model.encode_text(defect_pos_tokens)
            defect_pos_features = defect_pos_features / defect_pos_features.norm(dim=-1, keepdim=True)
            defect_pos_sim = (image_features @ defect_pos_features.T).mean().item()

            defect_neg_tokens = self.tokenizer(self.defect_negative).to(self.device)
            defect_neg_features = self.model.encode_text(defect_neg_tokens)
            defect_neg_features = defect_neg_features / defect_neg_features.norm(dim=-1, keepdim=True)
            defect_neg_sim = (image_features @ defect_neg_features.T).mean().item()

            defect_score = defect_pos_sim - defect_neg_sim

        raw_score = (
            -0.038554 * clarity_score +
            0.396935 * composition_score +
            0.641619 * defect_score
        )

        clarity_score_pct = (-0.038554 * clarity_score) / raw_score
        composition_score_pct = ( 0.396935 * composition_score ) / raw_score
        defect_score_pct = (0.641619 * defect_score) / raw_score

        final_score = 171.740104 * raw_score + 6.025378
        final_score = np.clip(final_score, 1.0, 10.0)

        # Empirical ranges from test_clip_ranges.py
        # clarity_pos: 0.15-0.21, clarity_neg: 0.12-0.21
        # comp_pos: 0.16-0.20, comp_neg: 0.14-0.21
        # defect_pos: 0.17-0.22, defect_neg: 0.16-0.21

        # Empirical ranges for combined scores (pos - neg) from test_clip_ranges.py
        # clarity: min(pos) - max(neg) to max(pos) - min(neg) = -0.06 to 0.09
        # composition: -0.05 to 0.05
        # defect: -0.05 to 0.06
        def normalize(val, min_val, max_val):
            normalized = (val - min_val) / (max_val - min_val)
            return float(np.clip(normalized, 0.0, 1.0))

        return {
            'clarity_score_est': normalize(clarity_score, -0.06, 0.09),
            'clarity_score': clarity_score_pct,
            'composition_score_est': normalize(composition_score, -0.05, 0.05),
            'composition_score': composition_score_pct,
            'defect_score_est': normalize(defect_score, -0.05, 0.06),
            'defect_score': defect_score_pct,
            'aesthetic_overall': float(final_score),
        }
