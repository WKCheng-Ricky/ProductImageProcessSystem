"""
Embedding Generation Module

Generate image vectors using CLIP
Generate descriptions using BLIP

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-07
Project: Product Image Processing System
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import open_clip
import torch
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .model_cache import get_model_cache

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    embedding: np.ndarray       # Vector
    model_name: str             # Model used
    dimension: int              # Vector dimension
    normalized: bool            # Whether normalized


class EmbeddingGenerator:
    # Supported models (open_clip format)
    SUPPORTED_MODELS = {
        'clip-vit-base': ('ViT-B-32', 'openai'),
        'clip-vit-large': ('ViT-L-14', 'openai'),
        'clip-vit-large-336': ('ViT-L-14-336', 'openai'),
        'laion-vit-b-32': ('ViT-B-32', 'laion2b_s34b_b79k'),
        'laion-vit-h-14': ('ViT-H-14', 'laion2b_s32b_b79k'),
    }

    def __init__(
        self,
        model_name: str = None,
        device: str = 'auto',
        normalize: bool = True,
    ):
        """
        Initialize embedding generator

        Args:
            model_name: Model name (see SUPPORTED_MODELS)
            device: 'cpu', 'cuda', or 'auto'
            normalize: Whether to L2 normalize
        """
        # Use environment variables if model_name not provided
        if model_name is None:
            env_model = os.getenv('CLIP_MODEL')
            env_pretrained = os.getenv('CLIP_PRETRAINED')
            if env_model and env_pretrained:
                # Check if this combination exists in SUPPORTED_MODELS
                for key, (arch, pretrained) in self.SUPPORTED_MODELS.items():
                    if arch == env_model and pretrained == env_pretrained:
                        model_name = key
                        break
                else:
                    # Not in SUPPORTED_MODELS, use direct format
                    model_name = f"{env_model}:{env_pretrained}"
            else:
                # Fallback to default
                model_name = 'clip-vit-large'

        self.model_name = model_name
        self.normalize = normalize

        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Lazy load
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._dimension = None

    def _load_model(self):
        """Load model using shared cache"""
        if self._model is not None:
            return

        try:
            # Get model configuration
            if self.model_name in self.SUPPORTED_MODELS:
                model_arch, pretrained = self.SUPPORTED_MODELS[self.model_name]
            elif ':' in self.model_name and self.model_name.startswith('hf-hub:'):
                # Support direct model strings like 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
                model_arch = self.model_name
                pretrained = None
            elif ':' in self.model_name:
                # Support format like 'ViT-B-32:laion2b_s34b_b79k'
                parts = self.model_name.split(':', 1)
                model_arch = parts[0]
                pretrained = parts[1]
            else:
                # Assume it's a model architecture name
                model_arch = self.model_name
                pretrained = None

            # Use shared model cache
            cache = get_model_cache()
            self._model, self._preprocess, self._tokenizer, self._dimension = cache.get_clip_model(
                model_arch,
                pretrained=pretrained,
                device=self.device,
            )

            logger.info(f"Model ready. Device: {self.device}, Dimension: {self._dimension}")

        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            raise ImportError(
                "Please install open_clip and torch: "
                "pip install open_clip_torch torch"
            )

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        self._load_model()
        return self._dimension

    def generate(self, img: Image.Image) -> EmbeddingResult:
        self._load_model()

        # Ensure RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Preprocessing
        image_input = self._preprocess(img).unsqueeze(0).to(self.device)

        # Generate embedding
        with torch.no_grad():
            features = self._model.encode_image(image_input)

        embedding = features.cpu().numpy().squeeze()

        # Normalize
        if self.normalize:
            embedding = embedding / np.linalg.norm(embedding)

        return EmbeddingResult(
            embedding=embedding,
            model_name=self.model_name,
            dimension=len(embedding),
            normalized=self.normalize,
        )

    def generate_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings in batch

        Args:
            images: List of images
            batch_size: Batch size
            show_progress: Whether to show progress

        Returns:
            List of EmbeddingResult
        """
        self._load_model()

        results = []

        # Prepare iterator
        if show_progress and tqdm is not None:
            iterator = tqdm(range(0, len(images), batch_size), desc="Generating embeddings")
        else:
            iterator = range(0, len(images), batch_size)

        for i in iterator:
            batch_images = images[i:i + batch_size]

            # Convert to RGB
            batch_images = [
                img.convert('RGB') if img.mode != 'RGB' else img
                for img in batch_images
            ]

            # Preprocessing - stack preprocessed images
            image_inputs = torch.stack([
                self._preprocess(img) for img in batch_images
            ]).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                features = self._model.encode_image(image_inputs)

            embeddings = features.cpu().numpy()

            # Normalize
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms

            # Wrap results
            for embedding in embeddings:
                results.append(EmbeddingResult(
                    embedding=embedding,
                    model_name=self.model_name,
                    dimension=len(embedding),
                    normalized=self.normalize,
                ))

        return results

    def generate_from_path(self, image_path: str) -> EmbeddingResult:
        """
        Generate embedding from file path

        Args:
            image_path: Image file path

        Returns:
            EmbeddingResult
        """
        img = Image.open(image_path)
        return self.generate(img)

    def similarity(
        self,
        embedding1: Union[np.ndarray, EmbeddingResult],
        embedding2: Union[np.ndarray, EmbeddingResult],
    ) -> float:
        """
        Calculate similarity between two embeddings (cosine similarity)

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score from -1 to 1
        """
        vec1 = embedding1.embedding if isinstance(embedding1, EmbeddingResult) else embedding1
        vec2 = embedding2.embedding if isinstance(embedding2, EmbeddingResult) else embedding2

        # Ensure normalized
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        return float(np.dot(vec1, vec2))

    def batch_similarity(
        self,
        query: Union[np.ndarray, EmbeddingResult],
        candidates: List[Union[np.ndarray, EmbeddingResult]],
    ) -> np.ndarray:
        """
        Calculate similarity between one query and multiple candidates

        Args:
            query: Query embedding
            candidates: List of candidate embeddings

        Returns:
            Similarity array
        """
        query_vec = query.embedding if isinstance(query, EmbeddingResult) else query
        query_vec = query_vec / np.linalg.norm(query_vec)

        candidate_vecs = np.array([
            c.embedding if isinstance(c, EmbeddingResult) else c
            for c in candidates
        ])

        # Normalize
        norms = np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
        candidate_vecs = candidate_vecs / norms

        # Calculate similarity
        similarities = np.dot(candidate_vecs, query_vec)

        return similarities


class TextImageEmbeddingGenerator(EmbeddingGenerator):
    """Text-Image embedding generator (CLIP specific)"""

    def generate_text_embedding(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for text

        Args:
            text: Input text

        Returns:
            EmbeddingResult
        """
        self._load_model()

        # Tokenize text
        text_tokens = self._tokenizer([text]).to(self.device)

        # Generate embedding
        with torch.no_grad():
            features = self._model.encode_text(text_tokens)

        embedding = features.cpu().numpy().squeeze()

        # Normalize
        if self.normalize:
            embedding = embedding / np.linalg.norm(embedding)

        return EmbeddingResult(
            embedding=embedding,
            model_name=self.model_name,
            dimension=len(embedding),
            normalized=self.normalize,
        )

    def text_image_similarity(
        self,
        text: str,
        img: Image.Image
    ) -> float:
        """
        Calculate similarity between text and image

        Args:
            text: Input text
            img: Input image

        Returns:
            Similarity score
        """
        text_emb = self.generate_text_embedding(text)
        img_emb = self.generate(img)

        return self.similarity(text_emb, img_emb)

    def search_by_text(
        self,
        query_text: str,
        images: List[Image.Image],
        top_k: int = 10,
    ) -> List[tuple]:
        """
        Search for similar images using text

        Args:
            query_text: Search text
            images: List of images
            top_k: Return top k results

        Returns:
            List of (index, similarity) tuples
        """
        # Generate text embedding
        text_emb = self.generate_text_embedding(query_text)

        # Generate image embeddings in batch
        img_embs = self.generate_batch(images)

        # Calculate similarity
        similarities = self.batch_similarity(text_emb, img_embs)

        # Sort
        indices = np.argsort(similarities)[::-1][:top_k]

        results = [(int(idx), float(similarities[idx])) for idx in indices]

        return results

if __name__ == "__main__":
    generator = EmbeddingGenerator()
    print("Embedding Generator module loaded successfully.")
