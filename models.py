from pydantic import BaseModel
from typing import Optional, List

class AiResponse(BaseModel):
    trash_detected: bool
    is_fake: Optional[bool]
    is_indoor: Optional[bool]
    trash_type: Optional[str]
    volume_estimate: Optional[str]
    ai_analysis: Optional[str]
    severity_score: Optional[float]

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