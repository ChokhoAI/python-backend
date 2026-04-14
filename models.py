from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List

class AiResponse(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda field_name: ''.join(
            word.capitalize() if i > 0 else word
            for i, word in enumerate(field_name.split('_'))
        ),
        populate_by_name=True
    )
    
    trash_detected: bool = False
    is_fake: bool = False
    is_indoor: bool = False
    trash_type: str = "OTHER"
    volume_estimate: str = "SMALL"
    ai_analysis: str = "No analysis"
    severity_score: float = 1.0

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