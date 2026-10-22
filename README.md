# Real vs AI

Flask application that classifies an uploaded JPG, PNG, or WEBP image as
`Real` or `AI Generated` using the MobileNetV2 model in
`models/real_vs_ai_mobilenetv2_pipeline.pkl`.

## Run locally

```bash
python --version
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
flask --app app run --debug
```

Open <http://127.0.0.1:5000> and upload an image. 
Uploads are validated, analyzed in memory, and are not stored by the application.

## API

`POST /analyze` accepts a multipart form field named `image` and returns:

```json
{
  "success": true,
  "result": "AI Generated or Real",
  "confidence": 94.27,
  "probabilities": {
    "real": 5.73,
    "ai_generated": 94.27
  }
}
```

Set `MODEL_PATH` to override the default model location.