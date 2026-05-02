# 🌾 Paddy Disease Detector

AI-powered rice leaf disease detection — FastAPI backend + HTML frontend.

## Folder Structure

```
paddy-disease-detector/
├── backend/
│   ├── main.py              ← FastAPI application
│   └── requirements.txt     ← Python dependencies
├── frontend/
│   └── index.html           ← Drag-and-drop UI (open in browser)
├── model/
│   └── best_model.pth       ← ⬅ PLACE YOUR MODEL HERE
└── README.md
```

---

## Quick Start

### 1. Place your model
```
cp /path/to/best_model.pth  paddy-disease-detector/model/best_model.pth
```

### 2. Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Start the API server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the frontend
Just open `frontend/index.html` in any browser — no server needed.

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Health ping |
| GET | `/health` | Model status + class list |
| POST | `/predict` | Upload image → prediction |

### POST /predict — Example Response
```json
{
  "predicted_class": "brown_spot",
  "label": "Brown Spot",
  "confidence": 94.37,
  "is_healthy": false,
  "top3": [
    { "class": "brown_spot",  "label": "Brown Spot",       "confidence": 94.37 },
    { "class": "blast",       "label": "Blast",            "confidence":  4.12 },
    { "class": "normal",      "label": "Normal / Healthy", "confidence":  1.51 }
  ]
}
```

---

## Supported Disease Classes (10)

| Class | Label |
|-------|-------|
| bacterial_leaf_blight | Bacterial Leaf Blight |
| bacterial_leaf_streak | Bacterial Leaf Streak |
| bacterial_panicle_blight | Bacterial Panicle Blight |
| blast | Blast |
| brown_spot | Brown Spot |
| dead_heart | Dead Heart |
| downy_mildew | Downy Mildew |
| hispa | Hispa |
| normal | Normal / Healthy |
| tungro | Tungro |

---

## Adapting to Your Model Architecture

If you used a different backbone (ResNet, VGG, ViT, etc.) open `backend/main.py`
and modify the `load_model()` function. Example for ResNet-50:

```python
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
```

If your checkpoint was saved with a `timm` EfficientNet backbone, install `timm` as well:

```bash
cd backend
pip install -r requirements.txt
```

This backend now supports the matching `timm` EfficientNet wrapper with `meta_mlp` and `head` modules. If your checkpoint uses a different custom architecture, update `backend/main.py` to match the model used during training.

The rest of the code (preprocessing, inference, API) stays the same.

---

## Test the API with curl
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/leaf_image.jpg"
```
