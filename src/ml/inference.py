"""
Keras Ensemble Inference Engine
================================
Loads the EfficientNet-B4 and ViT Keras models, runs ensemble
inference, and returns multi-label predictions with confidence scores.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
IMG_SIZE = (224, 224)           # Must match training resolution

# Default class names — overridden by model_meta.json if it exists
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def _load_model_meta() -> tuple[list[str], tuple[int, int]]:
    """Load class names and image size from training metadata."""
    import json
    meta_path = MODELS_DIR / "model_meta.json"
    names = CLASS_NAMES
    size = IMG_SIZE
    if meta_path.is_file():
        with open(meta_path) as f:
            meta = json.load(f)
        names = meta.get("class_names", names)
        raw_size = meta.get("img_size", list(size))
        size = tuple(raw_size)
    return names, size


CLASS_NAMES, IMG_SIZE = _load_model_meta()

# Ensemble weights  (equal by default — tune after validation)
EFFICIENTNET_WEIGHT = 0.5
VIT_WEIGHT = 0.5

# Confidence threshold for positive prediction
CONFIDENCE_THRESHOLD = 0.5


class KerasEnsemble:
    """Loads and runs the EfficientNet-B4 + ViT Keras ensemble."""

    def __init__(
        self,
        efficientnet_path: Optional[str] = None,
        vit_path: Optional[str] = None,
    ):
        eff_path = efficientnet_path or str(MODELS_DIR / "efficientnet_b4.keras")
        v_path = vit_path or str(MODELS_DIR / "vit.keras")

        self._eff_keras = None
        self._vit_keras = None

        # --- EfficientNet ---
        if os.path.isfile(eff_path):
            self._eff_keras = self._load_keras_model(eff_path, "EfficientNet-B4")

        # --- ViT ---
        if os.path.isfile(v_path):
            self._vit_keras = self._load_keras_model(v_path, "ViT")

        if not self._has_any_model():
            print(
                "[⚠] No models found (.keras). Inference will return "
                "dummy predictions. Train models first: "
                "python -m src.ml.train --data_dir ./data"
            )

    # ---------------------------------------------------------------
    # Keras model loading
    # ---------------------------------------------------------------
    @staticmethod
    def _load_keras_model(path: str, name: str):
        """Load a .keras model with custom objects."""
        try:
            import tensorflow as tf
            # Import custom classes so Keras can deserialise them
            from src.ml.train import (          # noqa: F401
                ConvStem, StochasticDepth, TransformerBlock,
                WarmupCosineDecay, AddPositionalEmbedding,
            )
            custom_objects = {
                "WarmupCosineDecay": WarmupCosineDecay,
                "ConvStem": ConvStem,
                "StochasticDepth": StochasticDepth,
                "TransformerBlock": TransformerBlock,
                "AddPositionalEmbedding": AddPositionalEmbedding,
            }
            model = tf.keras.models.load_model(path, custom_objects=custom_objects)
            print(f"[✓] {name} Keras model loaded from {path}")
            return model
        except Exception as exc:
            print(f"[⚠] Failed to load {name} Keras model: {exc}")
            return None

    def _has_any_model(self) -> bool:
        return any([self._eff_keras, self._vit_keras])

    # ---------------------------------------------------------------
    # Pre-processing
    # ---------------------------------------------------------------
    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Resize, apply CLAHE, and normalise for the ensemble.

        Mirrors the training preprocessing exactly:
        1. Convert to uint8 RGB
        2. Resize to IMG_SIZE
        3. CLAHE contrast enhancement (chest-X-ray-specific)
        4. Cast to float32 (keep [0, 255] — EfficientNet normalises
           internally; ViT BatchNorm handles raw pixel range)

        Parameters
        ----------
        image : np.ndarray
            A 2-D (H, W) greyscale or 3-D (H, W, 3) image.

        Returns
        -------
        np.ndarray
            Shape ``(1, H, W, 3)`` float32.
        """
        # Ensure 3 channels
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.concatenate([image] * 3, axis=-1)

        # Scale to [0, 255] uint8 if needed
        if image.dtype != np.uint8:
            img_min, img_max = image.min(), image.max()
            if img_max - img_min > 0:
                image = (image - img_min) / (img_max - img_min)
            image = (image * 255).astype(np.uint8)

        # Resize
        image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

        # CLAHE enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        image = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

        # Keep [0, 255] range — EfficientNet has built-in Rescaling /
        # Normalization layers; ViT's ConvStem BatchNorm handles it.
        # NOTE: preprocess_input() is a no-op in Keras 3, so training
        # also used [0, 255] float input.
        image = image.astype(np.float32)

        return np.expand_dims(image, axis=0)

    # ---------------------------------------------------------------
    # Ensemble prediction
    # ---------------------------------------------------------------
    def predict(self, image: np.ndarray) -> dict:
        """Run ensemble inference on a preprocessed or raw image.

        Returns
        -------
        dict
            {
              "predictions": [{"label": "PNEUMONIA", "confidence": 0.982}, ...],
              "raw_scores": [float, ...],
              "top_finding": "PNEUMONIA",
              "top_confidence": 0.982,
            }
        """
        img = self.preprocess(image)

        scores_list: list[np.ndarray] = []

        if self._eff_keras is not None:
            eff_scores = self._eff_keras.predict(img, verbose=0)
            scores_list.append(eff_scores * EFFICIENTNET_WEIGHT)

        if self._vit_keras is not None:
            vit_scores = self._vit_keras.predict(img, verbose=0)
            scores_list.append(vit_scores * VIT_WEIGHT)

        if scores_list:
            # Weighted average
            total_weight = sum(
                w for s, w in zip(scores_list, [EFFICIENTNET_WEIGHT, VIT_WEIGHT])
                if s is not None
            )
            combined = sum(scores_list) / total_weight if total_weight else sum(scores_list)
            combined = combined.squeeze()  # (NUM_CLASSES,)
        else:
            # Dummy fallback — random scores for dev / demo
            combined = np.random.rand(len(CLASS_NAMES)).astype(np.float32) * 0.3

        # Build structured output
        predictions = []
        for idx, (label, score) in enumerate(zip(CLASS_NAMES, combined)):
            if score >= CONFIDENCE_THRESHOLD:
                predictions.append({"label": label, "confidence": round(float(score), 4)})

        # Sort by confidence descending
        predictions.sort(key=lambda p: p["confidence"], reverse=True)

        top_idx = int(np.argmax(combined))
        return {
            "predictions": predictions,
            "raw_scores": [round(float(s), 4) for s in combined],
            "top_finding": CLASS_NAMES[top_idx],
            "top_confidence": round(float(combined[top_idx]), 4),
        }


# ---------------------------------------------------------------------------
# Module-level singleton for fast reuse across requests
# ---------------------------------------------------------------------------
_ensemble: Optional[KerasEnsemble] = None


def get_ensemble() -> KerasEnsemble:
    """Return a cached ensemble instance (lazy-loaded)."""
    global _ensemble
    if _ensemble is None:
        _ensemble = KerasEnsemble()
    return _ensemble
