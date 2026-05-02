from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
try:
    import timm
except ImportError:
    timm = None
from PIL import Image
import io
import os

app = FastAPI(title="Paddy Disease Detector API", version="1.0.0")

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Disease Classes (10 classes from Paddy Disease dataset) ──────────────────
CLASS_NAMES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]

CLASS_LABELS = {
    "bacterial_leaf_blight": "Bacterial Leaf Blight",
    "bacterial_leaf_streak": "Bacterial Leaf Streak",
    "bacterial_panicle_blight": "Bacterial Panicle Blight",
    "blast": "Blast",
    "brown_spot": "Brown Spot",
    "dead_heart": "Dead Heart",
    "downy_mildew": "Downy Mildew",
    "hispa": "Hispa",
    "normal": "Normal / Healthy",
    "tungro": "Tungro",
}

NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best_model.pth")

# ── Image Preprocessing ───────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── Model Loading ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def strip_prefix(state_dict, prefix: str):
    return {
        k[len(prefix):] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }


def normalize_state_dict(state_dict):
    # Remove DDP / DataParallel prefixes.
    if all(k.startswith("module.") for k in state_dict):
        state_dict = strip_prefix(state_dict, "module.")

    return state_dict


def assert_supported_checkpoint(state_dict):
    if any(k.startswith("meta_mlp") for k in state_dict) and not any(k.startswith("head.") for k in state_dict):
        raise ValueError(
            "Checkpoint contains a custom metadata block without a matching head. "
            "Please adapt backend/main.py to the exact model architecture used during training."
        )
    if any(k.startswith("head.") for k in state_dict) and not (any(k.startswith("meta_mlp") for k in state_dict) or any(k.startswith("classifier.") for k in state_dict)):
        raise ValueError(
            "Checkpoint contains an unsupported custom head format. "
            "Please adapt backend/main.py to the exact model architecture used during training."
        )


def build_torchvision_efficientnet():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


class EfficientNetWithMeta(nn.Module):
    def __init__(self, variant, state_dict):
        super().__init__()
        if timm is None:
            raise ImportError(
                "The saved checkpoint appears to use a timm EfficientNet backbone. "
                "Install timm with `pip install timm` or save a torchvision-compatible checkpoint."
            )

        self.backbone = timm.create_model(variant, pretrained=False, num_classes=0)
        self.meta_dim = state_dict["meta_mlp.0.weight"].shape[1]

        self.meta_mlp = nn.Sequential(
            nn.Linear(self.meta_dim, state_dict["meta_mlp.0.weight"].shape[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(state_dict["meta_mlp.0.weight"].shape[0], state_dict["meta_mlp.3.weight"].shape[0]),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(state_dict["head.1.weight"].shape[1], state_dict["head.1.weight"].shape[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(state_dict["head.1.weight"].shape[0], state_dict["head.4.weight"].shape[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(state_dict["head.4.weight"].shape[0], state_dict["head.7.weight"].shape[0]),
        )

    def forward(self, x, meta=None):
        x = self.backbone(x)
        if meta is None:
            meta = x.new_zeros(x.size(0), self.meta_dim)
        return self.head(torch.cat([x, self.meta_mlp(meta)], dim=1))


def detect_timm_efficientnet_variant(state_dict):
    # A few timm EfficientNet variants share the same early channel count.
    # We inspect a known block output shape to choose the right variant.
    key = "blocks.2.0.conv_pwl.weight"
    if key not in state_dict:
        key = "backbone.blocks.2.0.conv_pwl.weight"
    if key not in state_dict:
        return "efficientnet_b0"

    out_channels = state_dict[key].shape[0]
    variant_map = {
        40: "efficientnet_b0",
        48: "efficientnet_b2",
        56: "efficientnet_b4",
        64: "efficientnet_b5",
        72: "efficientnet_b6",
        80: "efficientnet_b7",
    }
    return variant_map.get(out_channels, "efficientnet_b0")


def build_timm_efficientnet(variant="efficientnet_b0"):
    if timm is None:
        raise ImportError(
            "The saved checkpoint appears to use a timm EfficientNet backbone. "
            "Install timm with `pip install timm` or save a torchvision-compatible checkpoint."
        )

    return timm.create_model(variant, pretrained=False, num_classes=NUM_CLASSES)


def build_timm_efficientnet_with_meta(state_dict):
    variant = detect_timm_efficientnet_variant(state_dict)
    return EfficientNetWithMeta(variant, state_dict)


def load_model():
    """
    Loads EfficientNet with a custom classifier head.
    Modify the architecture here if your training used a different backbone.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please place best_model.pth inside the model/ folder."
        )

    state = torch.load(MODEL_PATH, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError("Unexpected model checkpoint format. Expected a state_dict-like dict.")

    state = normalize_state_dict(state)
    assert_supported_checkpoint(state)

    if any(k.startswith("meta_mlp") for k in state) and any(k.startswith("head.") for k in state):
        model = build_timm_efficientnet_with_meta(state)
    elif any(k.startswith("backbone.") for k in state):
        state = strip_prefix(state, "backbone.")
        variant = detect_timm_efficientnet_variant(state)
        model = build_timm_efficientnet(variant)
    elif any(k.startswith("features.") for k in state):
        model = build_torchvision_efficientnet()
    elif any(k.startswith("conv_stem.") or k.startswith("blocks.") for k in state):
        variant = detect_timm_efficientnet_variant(state)
        model = build_timm_efficientnet(variant)
    else:
        model = build_torchvision_efficientnet()

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


try:
    model = load_model()
    print(f"Model loaded successfully on {device}")
except FileNotFoundError as e:
    print(f"Model not found: {e}")
    model = None
except Exception as e:
    print(f"Model load error: {e}")
    model = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Paddy Disease Detector API is running 🌾"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure best_model.pth is in the model/ folder.",
        )

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPEG or PNG image.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    # Preprocess
    tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze()

    confidence, predicted_idx = torch.max(probabilities, 0)
    predicted_class = CLASS_NAMES[predicted_idx.item()]

    # Build top-3 results
    top3_values, top3_indices = torch.topk(probabilities, 3)
    top3 = [
        {
            "class": CLASS_NAMES[i.item()],
            "label": CLASS_LABELS[CLASS_NAMES[i.item()]],
            "confidence": round(v.item() * 100, 2),
        }
        for v, i in zip(top3_values, top3_indices)
    ]

    return JSONResponse({
        "predicted_class": predicted_class,
        "label": CLASS_LABELS[predicted_class],
        "confidence": round(confidence.item() * 100, 2),
        "is_healthy": predicted_class == "normal",
        "top3": top3,
    })
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
