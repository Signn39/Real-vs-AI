from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from backend.image_validation import ImageValidationError, prepare_image
from backend.model_service import ModelLoadError, ModelService


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "real_vs_ai_mobilenetv2_pipeline.pkl"


def create_app(model_service: ModelService | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 + 1024
    service = model_service or ModelService(
        Path(os.environ.get("MODEL_PATH", str(MODEL_PATH)))
    )

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/analyze")
    def analyze() -> tuple[Any, int] | Any:
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return _error("MISSING_IMAGE", "Please select an image.", 400)

        try:
            image = prepare_image(upload.stream)
            result = service.predict(image)
        except ImageValidationError as error:
            return _error(error.code, error.message, error.status_code)
        except ModelLoadError as error:
            app.logger.error("Image analysis model is unavailable: %s", error)
            return _error(
                "MODEL_UNAVAILABLE",
                "The image analysis service is temporarily unavailable.",
                503,
            )
        except (RuntimeError, ValueError) as error:
            app.logger.exception("Image analysis failed: %s", error)
            return _error(
                "ANALYSIS_FAILED",
                "The image could not be analyzed.",
                500,
            )

        return jsonify({"success": True, **result})

    @app.errorhandler(413)
    def request_too_large(_: Exception) -> tuple[Any, int]:
        return _error(
            "FILE_TOO_LARGE",
            "Please upload an image no larger than 10 MB.",
            413,
        )

    return app


def _error(code: str, message: str, status_code: int) -> tuple[Any, int]:
    return jsonify({"success": False, "error": {"code": code, "message": message}}), status_code


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
