import base64
import io
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# Add classifier directory to path BEFORE importing local modules
CLASSIFIER_DIR = Path(__file__).parent
sys.path.insert(0, str(CLASSIFIER_DIR))

from src.inference import HanziClassifier
from src.utils import load_config, get_device


# Initialize classifier on startup
classifier: Optional[HanziClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
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


# Initialize API App
app = FastAPI(
    title="Hanzi Classifier MCP Server",
    description="MCP server for Hanzi character classification",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response models
class ClassifyRequest(BaseModel):
    """Request model for image classification."""
    image_base64: str
    return_confidence: bool = True


class ClassifyResponse(BaseModel):
    """Response model for classification results."""
    predicted_class: str
    confidence: float
    all_confidences: dict[str, float]


@app.post("/classify", response_model=ClassifyResponse)
async def classify_image(request: ClassifyRequest) -> ClassifyResponse:
    """Classify a Hanzi character image.
    
    Args:
        request: Contains base64-encoded image and options
        
    Returns:
        Classification results with predicted class and confidence scores
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "classifier_loaded": classifier is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.get("/classes")
async def get_classes():
    """Get available class names."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    return {
        "classes": classifier.class_names,
        "num_classes": len(classifier.class_names)
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )