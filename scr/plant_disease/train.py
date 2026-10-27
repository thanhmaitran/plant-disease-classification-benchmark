"""Training routines for both models."""

from __future__ import annotations

import time

import numpy as np

from .models import build_cnn, build_svm
from .preprocessing import extract_hog, normalise


def train_hog_svm(x_train, y_train, cfg: dict, seed: int):
    t0 = time.time()
    feats = extract_hog(
        x_train, cfg["orientations"], cfg["pixels_per_cell"], cfg["cells_per_block"]
    )
    model = build_svm(cfg["c"], cfg["class_weight"], seed)
    model.fit(feats, y_train)
    return model, time.time() - t0


def train_cnn(x_train, y_train, x_val, y_val, cfg: dict, image_size: int, seed: int):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    tf.random.set_seed(seed)
    n_classes = int(len(np.unique(y_train)))
    aug = cfg["augmentation"]
    gen = ImageDataGenerator(
        rotation_range=aug["rotation_range"],
        width_shift_range=aug["width_shift_range"],
        height_shift_range=aug["height_shift_range"],
        horizontal_flip=aug["horizontal_flip"],
        zoom_range=aug["zoom_range"],
        fill_mode="nearest",
    )
    xt, xv = normalise(x_train), normalise(x_val)
    yt = keras.utils.to_categorical(y_train, n_classes)
    yv = keras.utils.to_categorical(y_val, n_classes)
    model = build_cnn(image_size, n_classes, cfg["learning_rate"])
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg["early_stopping_patience"],
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]
    t0 = time.time()
    hist = model.fit(
        gen.flow(xt, yt, batch_size=cfg["batch_size"], seed=seed),
        epochs=cfg["epochs"], validation_data=(xv, yv),
        callbacks=callbacks, verbose=2,
    )
    return model, hist.history, time.time() - t0
