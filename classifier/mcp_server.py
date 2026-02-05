import base64
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import HTTPException
from pydantic import BaseModel
from fastmcp import FastMCP

from src.utils import get_device

# 1. SETUP PATHS
# Add classifier directory to path BEFORE importing local modules
CLASSIFIER_DIR = Path(__file__).parent
sys.path.insert(0, str(CLASSIFIER_DIR))

# Import your local module
from src.inference import HanziClassifier


# 3. GLOBAL STATE
# Define the variable
classifier: Optional[HanziClassifier] = None

# 4. DATA MODELS
class ClassifyRequest(BaseModel):
    image_base64: str
    return_confidence: bool = True

class ClassifyResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_confidences: dict[str, float]

# 5. LIFECYCLE EVENTS
@asynccontextmanager
async def lifespan(server: FastMCP):
    """Manage classifier lifecycle."""
    global classifier
    
    # Startup
    project_root = Path(__file__).parent
    config_path = project_root / "config.yaml"
    checkpoint_path = project_root / "checkpoints" / "hanzi_conv" / "best_model.pth"
    
    try:
        print(f"Loading classifier from {checkpoint_path}")
        classifier = HanziClassifier(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=get_device()
        )
        print("Classifier loaded successfully!")
    except Exception as e:
        print(f"Error loading classifier: {e}")
        raise
    
    yield  # App runs here
    
    # Shutdown
    print("Shutting down classifier")


# 2. INITIALIZE SERVER
mcp = FastMCP("Classifier", lifespan=lifespan)


# 6. TOOLS
@mcp.tool()
async def classify_image(request: ClassifyRequest) -> ClassifyResponse:
    """Classify a Hanzi character image."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier is still loading or failed to load.")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)

        # Classify
        pred_class, confidence, all_confidences = classifier.predict(
            image_data,
            return_confidence=request.return_confidence
        )
        
        return ClassifyResponse(
            predicted_class=pred_class,
            confidence=float(confidence),
            all_confidences={k: float(v) for k, v in all_confidences.items()}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error classifying image: {str(e)}")

@mcp.tool()
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "classifier_loaded": classifier is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

if __name__ == "__main__":
    # mcp.run() will enter the lifespan context automatically.
    mcp.run(transport="sse", host="0.0.0.0", port=8001)