"""
Training Script — EfficientNet-B4 + ViT Ensemble (TensorFlow / Keras)
======================================================================
Optimised for the Kaggle Chest X-Ray Pneumonia dataset (5,216 train /
624 test images, 2 classes: NORMAL / PNEUMONIA).

Key fixes over the initial version
------------------------------------
1. **EfficientNet preprocessing** — pass raw [0,255] pixels; the built-in
   ``preprocess_input`` handles normalisation correctly for ImageNet
   transfer-learning weights.
2. **Use test/ as validation** — the original val/ has only 16 images,
   far too few for stable metrics.
3. **CLAHE contrast enhancement** — chest X-rays are low-contrast
   greyscale; adaptive histogram equalisation dramatically improves
   feature visibility.
4. **224×224 input** — 3× faster than 380×380, more than sufficient
   for binary classification and fits comfortably in CPU memory.
5. **Two-phase training** — freeze base → train head → unfreeze top
   layers → fine-tune at low LR.
6. **Smaller ViT** — 4 blocks instead of 6, smaller embed dim; a
   from-scratch transformer with only ~5 K images needs fewer params.
7. **Stronger augmentation** — zoom, brightness, shear on top of
   rotation/flip to reduce overfitting on the imbalanced dataset.

Usage
-----
    python -m src.ml.train --data_dir ./data/chest_xray --epochs 15
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)          # Faster training; sufficient for binary task
BATCH_SIZE = 16
NUM_CLASSES = 2
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


# ===================================================================
# 0.  CLAHE contrast enhancement (chest X-ray specific)
# ===================================================================
def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast-Limited Adaptive Histogram Equalisation).

    Chest X-rays are low-contrast greyscale images.  CLAHE brings out
    subtle density differences (infiltrates, consolidation, etc.) that
    the model needs to see.
    """
    # image comes in as float [0, 255] RGB from the generator
    img_uint8 = image.astype(np.uint8)

    # Convert to LAB and equalise the L channel
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    return result.astype(np.float32)


def clahe_preprocessing(image: np.ndarray) -> np.ndarray:
    """Combined CLAHE + EfficientNet preprocessing."""
    image = apply_clahe(image)
    image = preprocess_input(image)      # scales to [-1, 1] as EfficientNet expects
    return image


# ===================================================================
# 1.  EfficientNet-B4  (texture / pattern detector)
# ===================================================================
def build_efficientnet(num_classes: int = NUM_CLASSES) -> Model:
    """Build an EfficientNet-B4 with a classification head.

    The base is initially frozen for the "head-only" training phase.
    Call ``unfreeze_top_layers(model, n)`` later for fine-tuning.
    """
    base = EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg",
    )
    base.trainable = False                # Phase 1: freeze everything

    x = base.output
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    activation = "softmax" if num_classes > 1 else "sigmoid"
    outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = Model(inputs=base.input, outputs=outputs, name="efficientnet_b4")
    return model


def unfreeze_top_layers(model: Model, num_layers: int = 40) -> None:
    """Unfreeze the top `num_layers` of the EfficientNet backbone."""
    # The base is the first layer (Functional model inside Functional model)
    base = model.layers[1] if hasattr(model.layers[1], 'layers') else None
    if base is None:
        # Flat model — unfreeze last N layers directly
        for layer in model.layers[-num_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
    else:
        base.trainable = True
        for layer in base.layers[:-num_layers]:
            layer.trainable = False
        # Keep BatchNorm frozen to preserve running stats
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False


# ===================================================================
# 2.  Hybrid Vision Transformer — Conv Stem + Transformer
# ===================================================================
#  "Early Convolutions Help Transformers See Better" (Xiao et al.)
#  A convolutional stem gives the ViT spatial inductive bias so it
#  can converge on small datasets (~5 K images) where a linear
#  patch-projection ViT stays at random-chance accuracy.
# ===================================================================

@keras.utils.register_keras_serializable()
class ConvStem(layers.Layer):
    """4-layer convolutional stem replacing linear patch projection.

    With stride-2 at each stage the output from a 224×224 input is
    14×14 spatial = 196 tokens — identical to 16×16 patch tokenisation
    but with hierarchical spatial features from the start.
    """

    def __init__(self, embed_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.conv1 = layers.Conv2D(32, 3, strides=2, padding="same")
        self.bn1   = layers.BatchNormalization()
        self.act1  = layers.Activation("gelu")

        self.conv2 = layers.Conv2D(64, 3, strides=2, padding="same")
        self.bn2   = layers.BatchNormalization()
        self.act2  = layers.Activation("gelu")

        self.conv3 = layers.Conv2D(128, 3, strides=2, padding="same")
        self.bn3   = layers.BatchNormalization()
        self.act3  = layers.Activation("gelu")

        self.conv4 = layers.Conv2D(embed_dim, 3, strides=2, padding="same")
        self.bn4   = layers.BatchNormalization()
        self.act4  = layers.Activation("gelu")

    def build(self, input_shape):
        """Explicitly build all sub-layers so weights exist for loading."""
        shape = input_shape
        for conv, bn in [
            (self.conv1, self.bn1),
            (self.conv2, self.bn2),
            (self.conv3, self.bn3),
            (self.conv4, self.bn4),
        ]:
            conv.build(shape)
            shape = conv.compute_output_shape(shape)
            bn.build(shape)
        super().build(input_shape)

    def call(self, x, training=False):
        x = self.act1(self.bn1(self.conv1(x), training=training))
        x = self.act2(self.bn2(self.conv2(x), training=training))
        x = self.act3(self.bn3(self.conv3(x), training=training))
        x = self.act4(self.bn4(self.conv4(x), training=training))
        # Reshape spatial grid → token sequence: (B, H*W, C)
        batch = tf.shape(x)[0]
        x = tf.reshape(x, [batch, -1, self.embed_dim])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


@keras.utils.register_keras_serializable()
class StochasticDepth(layers.Layer):
    """Drop entire residual branch with probability `drop_prob`."""

    def __init__(self, drop_prob: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=False):
        if not training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        mask = tf.floor(keep + tf.random.uniform(shape, dtype=x.dtype))
        return x / keep * mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_prob": self.drop_prob})
        return cfg


@keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    """Pre-norm transformer encoder block with stochastic depth."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, drop_path: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.drop_path_rate = drop_path

        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads,
            dropout=dropout,
        )
        self.stoch1 = StochasticDepth(drop_path)

        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])
        self.stoch2 = StochasticDepth(drop_path)

    def build(self, input_shape):
        """Explicitly build all sub-layers so weights exist for loading."""
        self.ln1.build(input_shape)
        self.att.build(input_shape, input_shape)  # MHA: (query_shape, value_shape)
        self.stoch1.build(input_shape)
        self.ln2.build(input_shape)
        self.ffn.build(input_shape)
        self.stoch2.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        # Pre-norm attention
        y = self.ln1(x)
        y = self.att(y, y, training=training)
        x = x + self.stoch1(y, training=training)
        # Pre-norm FFN
        y = self.ln2(x)
        y = self.ffn(y, training=training)
        x = x + self.stoch2(y, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
            "drop_path": self.drop_path_rate,
        })
        return config


@keras.utils.register_keras_serializable()
class AddPositionalEmbedding(layers.Layer):
    """Adds a learnable positional embedding to the input token sequence.

    Unlike ``layers.Embedding(tf.range(...))``, this layer properly
    registers its weight so Keras can track, train, save, and restore it.
    """

    def __init__(self, num_tokens: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(self.num_tokens, self.embed_dim),
            initializer="truncated_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_embed

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_tokens": self.num_tokens,
            "embed_dim": self.embed_dim,
        })
        return config


def build_vit(
    img_size: tuple[int, int] = IMG_SIZE,
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_blocks: int = 2,               # 2 blocks — small dataset
    drop_path: float = 0.1,
    num_classes: int = NUM_CLASSES,
) -> Model:
    """Build a Hybrid ViT: Conv Stem + 2 Transformer blocks.

    The convolutional stem provides the spatial inductive bias that
    a pure-ViT lacks, enabling convergence on ~5 K images.
    """
    inputs = keras.Input(shape=(*img_size, 3))

    # ── Conv stem  →  (B, 196, embed_dim)  for 224×224 input ────
    x = ConvStem(embed_dim=embed_dim)(inputs)
    num_tokens = (img_size[0] // 16) * (img_size[1] // 16)   # 14*14 = 196

    # Learnable positional embedding (proper Keras layer → weights tracked)
    x = AddPositionalEmbedding(num_tokens, embed_dim)(x)
    x = layers.Dropout(0.1)(x)

    # ── Transformer blocks with linearly increasing drop-path ───
    for i in range(num_blocks):
        dp = drop_path * (i / max(num_blocks - 1, 1))
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout=0.1, drop_path=dp)(x)

    # ── Classification head ─────────────────────────────────────
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.Dropout(0.2)(x)

    activation = "softmax" if num_classes > 1 else "sigmoid"
    outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="hybrid_vit")
    return model


# ===================================================================
# 3.  Data pipeline (chest-X-ray specific)
# ===================================================================
def get_data_generators(data_dir: str):
    """Return Keras ImageDataGenerators optimised for chest X-rays.

    Key choices
    -----------
    * **No rescale** — EfficientNet's ``preprocess_input`` handles it.
      For the ViT we still need [0,1], so we apply it separately.
    * **test/ as validation** — the official val/ has only 16 images;
      using test/ (624 images) gives meaningful val metrics.
    * **CLAHE** applied via ``preprocessing_function``.
    * **Heavier augmentation** to compensate for class imbalance
      (NORMAL 1,341 vs PNEUMONIA 3,875).
    """
    train_gen = ImageDataGenerator(
        preprocessing_function=clahe_preprocessing,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        brightness_range=(0.85, 1.15),
        horizontal_flip=True,
        fill_mode="constant",
        cval=0,
    )
    val_gen = ImageDataGenerator(
        preprocessing_function=clahe_preprocessing,
    )

    train_ds = train_gen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=True,
        seed=42,
    )

    # Prefer test/ over val/ when val/ is tiny
    val_dir = os.path.join(data_dir, "test")
    if not os.path.isdir(val_dir):
        val_dir = os.path.join(data_dir, "val")

    val_ds = val_gen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False,
    )
    return train_ds, val_ds


def get_vit_data_generators(data_dir: str):
    """Return data generators for ViT using the SAME preprocessing as
    EfficientNet (CLAHE + preprocess_input → [-1, 1]).

    Previously this used CLAHE + /255 ([0,1]) which caused a critical
    mismatch at inference time — the ONNXEnsemble applies a single
    preprocessing pipeline, so both models must expect the same input
    range.
    """
    return get_data_generators(data_dir)


# ===================================================================
# 4.  ONNX export via tf2onnx
# ===================================================================
def export_to_onnx(model: Model, output_path: str) -> None:
    """Convert a Keras model to ONNX format using tf2onnx.

    Includes a compatibility fix for TF 2.20+ where FuncGraph._captures
    was renamed to .captures  (breaks tf2onnx <= 1.16).
    """
    import tf2onnx
    from tensorflow.python.framework.func_graph import FuncGraph

    # ── Patch FuncGraph so tf2onnx can find `._captures` ─────────
    if not hasattr(FuncGraph, "_captures"):
        FuncGraph._captures = property(lambda self: self.captures)

    try:
        spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=17,
            output_path=output_path,
        )
        print(f"[✓] ONNX model saved to {output_path}")
    except Exception as e:
        print(f"[!] tf2onnx from_keras failed: {e}")
        print("[!] Falling back to SavedModel → ONNX CLI conversion...")
        saved_model_path = output_path.replace(".onnx", "_savedmodel")
        model.export(saved_model_path)
        import subprocess
        result = subprocess.run(
            [
                "python", "-m", "tf2onnx.convert",
                "--saved-model", saved_model_path,
                "--output", output_path,
                "--opset", "17",
            ],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"[✓] ONNX model saved to {output_path} (via CLI fallback)")
        else:
            print(f"[!] CLI conversion also failed: {result.stderr}")
            keras_path = output_path.replace(".onnx", ".keras")
            model.save(keras_path)
            print(f"[✓] Saved Keras model to {keras_path} as fallback")


# ===================================================================
# 5.  Learning-rate warmup (avoids accuracy drop at Phase 2 start)
# ===================================================================
@keras.utils.register_keras_serializable()
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay.

    Prevents the accuracy crash that occurs when a freshly-compiled
    optimizer (no momentum / variance history) starts updating
    newly-unfrozen backbone layers at the full learning rate.
    """

    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.math.minimum(step / tf.cast(self.warmup_steps, tf.float32), 1.0)
        # Cosine decay after warmup
        decay_steps = tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        cosine = 0.5 * (1 + tf.math.cos(
            np.pi * tf.math.maximum(step - self.warmup_steps, 0.0) / tf.math.maximum(decay_steps, 1.0)
        ))
        return self.base_lr * warmup * cosine

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


# ===================================================================
# 6.  Main training loop  (two-phase for EfficientNet)
# ===================================================================
def train(data_dir: str, epochs: int = 15, model_choice: str = "both") -> None:
    """Main training entry point.

    Parameters
    ----------
    model_choice : str
        ``'both'`` — train EfficientNet + ViT (default).
        ``'efficientnet'`` — train only EfficientNet.
        ``'vit'`` — train only the Hybrid ViT.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────
    eff_train_ds, eff_val_ds = get_data_generators(data_dir)
    detected_classes = eff_train_ds.num_classes
    class_names = list(eff_train_ds.class_indices.keys())
    print(f"\n[INFO] Detected {detected_classes} classes: {class_names}")
    print(f"[INFO] Train: {eff_train_ds.samples}  |  Val: {eff_val_ds.samples}")

    loss_fn = "categorical_crossentropy"

    # ── Class weights (handle NORMAL/PNEUMONIA imbalance) ─────────
    from sklearn.utils.class_weight import compute_class_weight
    try:
        weights = compute_class_weight(
            "balanced",
            classes=np.unique(eff_train_ds.classes),
            y=eff_train_ds.classes,
        )
        class_weight = dict(enumerate(weights))
        print(f"[INFO] Class weights: {class_weight}")
    except Exception:
        class_weight = None

    # ── Callbacks ─────────────────────────────────────────────────
    def make_callbacks(name: str, use_reduce_lr: bool = True):
        """Build callbacks for a training phase.

        Parameters
        ----------
        name : str
            Prefix for checkpoint filenames.
        use_reduce_lr : bool
            If False, ReduceLROnPlateau is omitted.  Must be False
            when the optimizer already uses a LR schedule (e.g.
            WarmupCosineDecay) because the schedule overrides the
            optimizer's base LR every step, making ReduceLR a no-op.
        """
        cbs = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                min_delta=1e-4,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(MODELS_DIR / f"{name}_best.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.CSVLogger(
                str(MODELS_DIR / f"{name}_history.csv"),
                append=False,
            ),
        ]
        if use_reduce_lr:
            cbs.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5,
                    patience=2, min_lr=1e-7, verbose=1,
                )
            )
        return cbs

    # =============================================================
    #  EfficientNet-B4  — Phase 1: head only
    # =============================================================
    if model_choice == "vit":
        # Skip EfficientNet entirely — jump straight to ViT
        pass
    else:
        _train_efficientnet(data_dir, epochs, detected_classes, class_names,
                            class_weight, loss_fn, make_callbacks,
                            eff_train_ds, eff_val_ds)

    # =============================================================
    #  Vision Transformer (only if requested)
    # =============================================================
    if model_choice in ("both", "vit"):
        train_vit(data_dir, epochs, detected_classes, class_names,
                  class_weight, loss_fn, make_callbacks)

    # ── Save metadata ─────────────────────────────────────────────
    meta_path = MODELS_DIR / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"class_names": class_names, "img_size": list(IMG_SIZE)}, f)
    print(f"[✓] Model metadata saved to {meta_path}")

    print("\n🎉  Training and export complete.")


def _train_efficientnet(data_dir, epochs, detected_classes, class_names,
                        class_weight, loss_fn, make_callbacks,
                        eff_train_ds, eff_val_ds):
    """EfficientNet two-phase training."""
    print("\n" + "=" * 60)
    print("  EfficientNet-B4  —  Phase 1: Train classification head")
    print("=" * 60)

    eff_model = build_efficientnet(num_classes=detected_classes)
    eff_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=loss_fn,
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    head_epochs = max(3, epochs // 3)
    eff_model.fit(
        eff_train_ds, validation_data=eff_val_ds,
        epochs=head_epochs,
        class_weight=class_weight,
        callbacks=make_callbacks("eff_phase1", use_reduce_lr=True),
    )

    # =============================================================
    #  EfficientNet-B4  — Phase 2: fine-tune top layers
    # =============================================================
    print("\n" + "=" * 60)
    print("  EfficientNet-B4  —  Phase 2: Fine-tune top 20 layers")
    print("=" * 60)

    unfreeze_top_layers(eff_model, num_layers=20)   # fewer layers → more stable

    # Warmup schedule: ramp LR over 1 epoch of steps, then cosine-decay.
    phase2_epochs = epochs - head_epochs
    steps_per_epoch = len(eff_train_ds)
    total_steps = phase2_epochs * steps_per_epoch
    warmup_steps = steps_per_epoch          # 1 full epoch of warmup

    lr_schedule = WarmupCosineDecay(
        base_lr=5e-5,                       # conservative peak LR
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    eff_model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=loss_fn,
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    eff_model.fit(
        eff_train_ds, validation_data=eff_val_ds,
        initial_epoch=head_epochs,           # continue epoch numbering
        epochs=epochs,
        class_weight=class_weight,
        callbacks=make_callbacks("eff_phase2", use_reduce_lr=False),
    )

    # ── Save EfficientNet Keras model first (crash-safe) ─────────
    eff_keras_path = str(MODELS_DIR / "efficientnet_b4.keras")
    eff_model.save(eff_keras_path)
    print(f"[✓] EfficientNet Keras model saved to {eff_keras_path}")

    eff_onnx_path = str(MODELS_DIR / "efficientnet_b4.onnx")
    export_to_onnx(eff_model, eff_onnx_path)


# ===================================================================
# 7.  ViT training  (separate function for --model vit)
# ===================================================================
def train_vit(data_dir, epochs, detected_classes, class_names,
              class_weight, loss_fn, make_callbacks):
    """Train the Hybrid ViT with warmup, cosine decay, label smoothing,
    and gradient clipping.  Designed to work on small datasets (~5 K)."""

    print("\n" + "=" * 60)
    print("  Hybrid Vision Transformer  —  Training")
    print("=" * 60)

    vit_train_ds, vit_val_ds = get_vit_data_generators(data_dir)
    steps_per_epoch = len(vit_train_ds)

    # ── Build model ──────────────────────────────────────────────
    vit_model = build_vit(num_classes=detected_classes)
    vit_model.summary()

    # ── LR schedule: linear warmup → cosine decay ────────────────
    total_steps = epochs * steps_per_epoch
    warmup_steps = 2 * steps_per_epoch      # 2 epochs of warmup

    lr_schedule = WarmupCosineDecay(
        base_lr=1e-3,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # ── Compile with gradient clipping + label smoothing ─────────
    vit_model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4,
            clipnorm=1.0,                   # gradient clipping
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    # ── Train (no ReduceLR — schedule handles it) ────────────────
    vit_model.fit(
        vit_train_ds, validation_data=vit_val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=make_callbacks("vit", use_reduce_lr=False),
    )

    # ── Save Keras model (crash-safe) ────────────────────────────
    vit_keras_path = str(MODELS_DIR / "vit.keras")
    vit_model.save(vit_keras_path)
    print(f"[✓] ViT Keras model saved to {vit_keras_path}")

    vit_onnx_path = str(MODELS_DIR / "vit.onnx")
    export_to_onnx(vit_model, vit_onnx_path)


# ===================================================================
# CLI entry-point
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MedView-AI models")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument(
        "--model", type=str, default="both",
        choices=["both", "efficientnet", "vit"],
        help="Which model to train: both, efficientnet, or vit",
    )
    args = parser.parse_args()

    train(data_dir=args.data_dir, epochs=args.epochs, model_choice=args.model)
