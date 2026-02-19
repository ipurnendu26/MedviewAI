"""
Grad-CAM Explainability Module (TensorFlow / Keras)
====================================================
Generates class-activation heatmaps that highlight which regions of a
chest X-ray contributed most to the model's prediction.

The implementation works with both EfficientNet-B4 and the custom ViT
built in ``train.py``.  For ONNX-only deployments (no TF model in
memory), a lightweight *approximation* based on the ONNX output is
provided.
"""

from __future__ import annotations

import io
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Try importing TensorFlow (optional at inference time)
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ===================================================================
# 1.  TensorFlow Grad-CAM  (used when the Keras model is in memory)
# ===================================================================

def grad_cam_tf(
    model: "keras.Model",
    image: np.ndarray,
    class_index: int,
    last_conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmap using a TF/Keras model.

    Parameters
    ----------
    model : keras.Model
        A compiled Keras model with at least one Conv2D layer.
    image : np.ndarray
        Pre-processed image of shape ``(1, H, W, 3)`` as float32.
    class_index : int
        Target class index for the heatmap.
    last_conv_layer_name : str, optional
        Name of the last convolutional layer. If ``None``, auto-detected.

    Returns
    -------
    np.ndarray
        Heatmap of shape ``(H, W)`` with values in ``[0, 1]``.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for grad_cam_tf")

    # Auto-detect last conv layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, (keras.layers.Conv2D,)):
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No Conv2D layer found in model")

    # Build a sub-model that returns conv outputs + predictions
    grad_model = keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_index]

    # Gradients of the class score w.r.t. the conv feature map
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalise
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


# ===================================================================
# 2.  Saliency-based approximation for ONNX-only deployments
# ===================================================================

def saliency_map_approx(
    raw_scores: list[float],
    image: np.ndarray,
    class_index: int,
) -> np.ndarray:
    """Generate a rough saliency-style heatmap without TF in memory.

    This uses a simple Sobel edge + intensity weighting as a
    placeholder.  Replace with ONNX-compatible Grad-CAM once a
    custom ONNX graph with intermediate outputs is available.

    Parameters
    ----------
    raw_scores : list[float]
        Model output scores (used for intensity scaling).
    image : np.ndarray
        Original greyscale image ``(H, W)`` or ``(H, W, 3)``.
    class_index : int
        Index of the target class.

    Returns
    -------
    np.ndarray
        Heatmap ``(H, W)`` in ``[0, 1]``.
    """
    if image.ndim == 3:
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[-1] == 3 else image[:, :, 0]
    else:
        grey = image.copy()

    grey = grey.astype(np.float32)

    # Sobel edge magnitude as a rough attention proxy
    grad_x = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Gaussian blur for smoother heatmap
    magnitude = cv2.GaussianBlur(magnitude, (15, 15), 0)

    # Normalise
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude /= mag_max

    # Scale by class confidence
    confidence = raw_scores[class_index] if class_index < len(raw_scores) else 0.5
    magnitude *= confidence

    return magnitude


# ===================================================================
# 3.  Overlay heatmap on original image
# ===================================================================

def overlay_heatmap(
    original: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a heatmap on the original image.

    Parameters
    ----------
    original : np.ndarray
        Original image (H, W) or (H, W, 3) in uint8 or float32.
    heatmap : np.ndarray
        Heatmap (H, W) in [0, 1].
    alpha : float
        Blending factor for the heatmap.
    colormap : int
        OpenCV colourmap constant.

    Returns
    -------
    np.ndarray
        Overlaid image (H, W, 3) as uint8.
    """
    # Convert heatmap to colour
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # Ensure original is uint8 RGB
    if original.dtype != np.uint8:
        orig_display = np.uint8(255 * np.clip(original, 0, 1))
    else:
        orig_display = original.copy()

    if orig_display.ndim == 2:
        orig_display = cv2.cvtColor(orig_display, cv2.COLOR_GRAY2BGR)

    # Blend
    overlay = cv2.addWeighted(orig_display, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


# ===================================================================
# 4.  Convenience: generate heatmap PNG bytes
# ===================================================================

def generate_heatmap_png(
    original_image: np.ndarray,
    raw_scores: list[float],
    class_index: int,
    keras_model: Optional[object] = None,
) -> bytes:
    """Generate a Grad-CAM / saliency heatmap and return PNG bytes.

    If a live Keras model is provided AND TensorFlow is available,
    true Grad-CAM is used.  Otherwise falls back to the Sobel-based
    saliency approximation.
    """
    # Decide method
    if keras_model is not None and TF_AVAILABLE:
        from src.ml.inference import IMG_SIZE as _INF_SIZE
        resized = cv2.resize(original_image, _INF_SIZE)
        if resized.ndim == 2:
            resized = np.stack([resized] * 3, axis=-1)
        # Scale to [0, 255] uint8 for CLAHE
        if resized.dtype != np.uint8:
            r_min, r_max = resized.min(), resized.max()
            if r_max - r_min > 0:
                resized = (resized - r_min) / (r_max - r_min)
            resized = (resized * 255).astype(np.uint8)
        # CLAHE contrast enhancement
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        resized = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        # Keep [0, 255] range — models handle normalisation internally
        preprocessed = resized.astype(np.float32)
        preprocessed = np.expand_dims(preprocessed, axis=0)
        heatmap = grad_cam_tf(keras_model, preprocessed, class_index)
    else:
        heatmap = saliency_map_approx(raw_scores, original_image, class_index)

    # Overlay
    overlay = overlay_heatmap(original_image, heatmap)

    # Encode to PNG bytes
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title(f"Grad-CAM — Class {class_index}")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
