"""Model definitions: HOG + SVM pipeline and the custom CNN."""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm(c=10.0, class_weight="balanced", seed=42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=c, gamma="scale",
                        class_weight=class_weight, random_state=seed)),
        ]
    )


def build_cnn(image_size=96, n_classes=3, learning_rate=1e-3):
    """Custom 3-block CNN trained from scratch (no pretrained weights)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(image_size, image_size, 3)),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2), layers.Dropout(0.25),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2), layers.Dropout(0.25),
            layers.Conv2D(96, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2), layers.Dropout(0.3),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
