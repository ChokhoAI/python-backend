import google.genai as genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def verify_image(original_img , cleaned_img):
    image1 = types.Part.from_bytes(
        data=original_img, mime_type="image/jpeg"
    )
    image2 = types.Part.from_bytes(
        data=cleaned_img, mime_type="image/jpeg"
    )

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            """You are a waste verification assistant. You will be given two images:
            1. Original complaint photo showing trash at a location 
            2. Verification photo taken after cleaning

            Analyze both images and respond in JSON format only, no markdown, no explanation outside JSON:
            {
            "is_cleaned": true or false,
            "reasoning": "brief explanation"
            }

            is_cleaned should be true only if the trash visible in the original photo is clearly removed or significantly reduced in the verification photo. If the images appear to be from completely different locations, set is_cleaned to false.""" , image1 , image2
        ]
    )
    return response.text