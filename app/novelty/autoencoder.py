"""
Autoencoder model for Android malware novelty detection.

This module only builds and compiles the TensorFlow/Keras autoencoder.
Training and prediction are handled elsewhere.
"""

from __future__ import annotations

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Dropout,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


class AutoencoderBuilder:
    """
    Builds a deep symmetric autoencoder for novelty detection.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        learning_rate: float = 1e-3,
        l2_lambda: float = 1e-4,
        dropout_rate: float = 0.20,
    ):

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate

    def build(self) -> Model:
        """
        Returns a compiled TensorFlow autoencoder.
        """

        inputs = Input(
            shape=(self.input_dim,),
            name="input_features"
        )

        # ---------------- Encoder ----------------

        x = Dense(
            256,
            activation="relu",
            kernel_regularizer=l2(self.l2_lambda)
        )(inputs)

        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(
            128,
            activation="relu",
            kernel_regularizer=l2(self.l2_lambda)
        )(x)

        x = BatchNormalization()(x)

        x = Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(self.l2_lambda)
        )(x)

        latent = Dense(
            self.latent_dim,
            activation="relu",
            name="latent_space"
        )(x)

        # ---------------- Decoder ----------------

        x = Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(self.l2_lambda)
        )(latent)

        x = BatchNormalization()(x)

        x = Dense(
            128,
            activation="relu",
            kernel_regularizer=l2(self.l2_lambda)
        )(x)

        x = BatchNormalization()(x)

        x = Dense(
            256,
            activation="relu",
            kernel_regularizer=l2(self.l2_lambda)
        )(x)

        outputs = Dense(
            self.input_dim,
            activation="linear",
            name="reconstruction"
        )(x)

        model = Model(
            inputs,
            outputs,
            name="AndroidNoveltyAutoencoder"
        )

        model.compile(
            optimizer=Adam(
                learning_rate=self.learning_rate
            ),
            loss="mse",
            metrics=["mae"]
        )

        return model


def build_autoencoder(
    input_dim: int,
    latent_dim: int = 32,
) -> Model:
    """
    Convenience function used by the training pipeline.
    """

    builder = AutoencoderBuilder(
        input_dim=input_dim,
        latent_dim=latent_dim,
    )

    return builder.build()


if __name__ == "__main__":

    model = build_autoencoder(470)

    model.summary()