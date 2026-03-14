from fastapi import FastAPI , File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated

app = FastAPI()

class ComplaintResponse(BaseModel):
    complaint_id : int
    user_id : int

    


app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)



@app.get("/")
async def root():
    return {"message" : "Welcome to Waste Detection Backend"}

@app.post("/detect")
async def get_image(complaint_id : int , user_id : int, image : UploadFile = File(...)):
    return