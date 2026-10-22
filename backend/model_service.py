from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class ModelLoadError(RuntimeError):
    """Raised when the configured model cannot be loaded or validated."""


class ModelService:
    """Loads the serialized Keras model and performs image classification."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model: Any | None = None
        self.class_names: tuple[str, str] = ("Real_Images", "AI_Images")
        self.target_size = (224, 224)

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import joblib
            from tensorflow import keras

            bundle = joblib.load(self.model_path)
            model_json = bundle["model_json"]
            weights = bundle["model_weights"]
            class_names = tuple(bundle["class_names"])
            target_size = tuple(bundle["target_size"])

            if len(class_names) != 2 or len(target_size) != 2:
                raise ValueError("Model metadata has an unsupported shape.")
            if class_names != ("Real_Images", "AI_Images"):
                raise ValueError("Model class order must be Real_Images, AI_Images.")
            if tuple(target_size) != self.target_size:
                raise ValueError(f"Expected target size {self.target_size}.")

            model = keras.models.model_from_json(model_json)
            model.set_weights(weights)
            output_shape = model.output_shape
            if output_shape[-1] != 1:
                raise ValueError("Expected a binary model with one output.")

            self._model = model
            self.class_names = (str(class_names[0]), str(class_names[1]))
            self.target_size = (int(target_size[0]), int(target_size[1]))
        except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
            raise ModelLoadError(
                f"Unable to load model from {self.model_path}: {error}"
            ) from error

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        self._load()
        assert self._model is not None

        batch = np.expand_dims(image, axis=0)
        # The serialized model starts with x / 127.5 - 1.0.
        prediction = np.asarray(self._model.predict(batch, verbose=0), dtype=float)
        if prediction.shape != (1, 1):
            raise ModelLoadError("Model returned an invalid prediction shape.")

        ai_probability = float(np.clip(prediction[0, 0], 0.0, 1.0))
        real_probability = 1.0 - ai_probability
        is_ai = ai_probability >= 0.5

        return {
            "result": "AI Generated" if is_ai else "Real",
            "confidence": round((ai_probability if is_ai else real_probability) * 100, 2),
            "probabilities": {
                "real": round(real_probability * 100, 2),
                "ai_generated": round(ai_probability * 100, 2),
            },
        }
