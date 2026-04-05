from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from collections import defaultdict
from sklearn.cluster import KMeans
from services.tsp import nearest_neighbor_tsp
from services.verify_image import verify_image

import requests
import json
import numpy as np


app = FastAPI()


class ComplaintResponse(BaseModel):
    complaint_id : int
    user_id : int


class ComplaintRequest(BaseModel):
    id : int
    latitude : float
    longitude : float

class RouteOptimizationRequest(BaseModel):
    complaints : List[ComplaintRequest]
    total_vehicles : int

class RouteResult(BaseModel):
    cluster_id : int
    complaint_ids : List[int] 

class RouteOptimzationResponse(BaseModel):
    routes : List[RouteResult]

class VerificationResponse(BaseModel):
    is_cleaned : bool
    reason : str

class VerificationRequest(BaseModel):
    original_img_url : str
    cleaned_img_url : str


app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)



@app.get("/")
async def root():
    return {"message" : "chokho-python-backend is running"}

# @app.post("/detect")
# async def get_image(complaint_id : int , user_id : int, image : UploadFile = File(...)):
#     return

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
    original_img = requests.get(requestModel.original_img_url).content
    cleaned_img = requests.get(requestModel.cleaned_img_url).content

    response = verify_image(original_img,cleaned_img)
    result = json.loads(response)

    return VerificationResponse(
        is_cleaned= result["is_cleaned"],
        reason= result["reasoning"]
    )