from fastapi import FastAPI , File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np
from services.tsp import nearest_neighbor_tsp

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