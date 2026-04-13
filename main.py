from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from collections import defaultdict
from sklearn.cluster import KMeans
from services.tsp import nearest_neighbor_tsp
from models import AiResponse, VerificationRequest ,VerificationResponse , RouteOptimizationRequest , RouteOptimzationResponse , RouteResult
from model_loader import load_model_startup, is_model_ready
import requests
import json
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chokho AI Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:3000",
                    "https://chokho-frontend.vercel.app",
                    "https://chokho-backend.onrender.com"
                    ],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

# Startup event: Load model when server starts
@app.on_event("startup")
async def startup_event():
    """Load YOLO model on server startup to avoid request delays."""
    try:
        logger.info("Server starting - initializing AI model...")
        load_model_startup()
        logger.info("Server ready to handle requests")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # Don't crash server, but log the error

@app.get("/")
async def root():
    return {"message" : "chokho-python-backend is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint that includes model readiness status."""
    return {
        "status": "healthy",
        "model_ready": is_model_ready(),
        "service": "Chokho AI Backend"
    }

@app.post("/analyze", response_model=AiResponse)
async def ai_analysis(image : UploadFile = File(...)):
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="AI model is still loading, please retry in a moment")
    
    from services.ai import complaint_check
    file_bytes = await image.read()
    
    try:
        result = await complaint_check(file_bytes)
        return result
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error processing image")

@app.post("/routes" , response_model= RouteOptimzationResponse)
async def route_optimization(request : RouteOptimizationRequest):
    if len(request.complaints) < request.total_vehicles:
        raise HTTPException(status_code=400, detail="Number of complaints must be lower than number of complaints")\
        
    k = request.total_vehicles
    model = KMeans(n_clusters=k, random_state= 42)

    coordinates = np.array([
        [complaint.latitude,complaint.longitude] for complaint in request.complaints
    ])

    ids = [complaint.id for complaint in request.complaints]

    model.fit(coordinates)
    labels = model.labels_

    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append({'id': ids[i], 'coords': coordinates[i]})

    routes = []

    for cluster_label, complaints in clusters.items():
        ordered_ids = nearest_neighbor_tsp(complaints=complaints)
        routes.append(RouteResult(
            cluster_id= cluster_label,
            complaint_ids= ordered_ids
        ))

    return RouteOptimzationResponse(routes=routes)

@app.post("/verify", response_model= VerificationResponse)
async def verify(requestModel : VerificationRequest):
    from services.ai import verify_image
    try:
        original_img = requests.get(requestModel.original_img_url, timeout=10).content
        cleaned_img = requests.get(requestModel.cleaned_img_url, timeout=10).content

        response = verify_image(original_img,cleaned_img)
        result = json.loads(response)

        return VerificationResponse(
            is_cleaned= result["is_cleaned"],
            reason= result["reasoning"]
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Request timeout while fetching images")
    except Exception as e:
        logger.error(f"Error in verify endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error processing verification")

if __name__ == "__main__":
    import uvicorn
    # Configuration optimized for Render free tier
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=10000,
        workers=1,  # Single worker for free tier stability
        timeout_keep_alive=5,
        timeout_notify=60  # Longer timeout for requests
    )