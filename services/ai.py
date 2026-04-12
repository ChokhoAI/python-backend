import google.genai as genai
from google.genai import types
import os , json
from dotenv import load_dotenv
from services.detection import detect_trash
from models import AiResponse

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

async def verify_image(original_img , cleaned_img):
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

async def complaint_check(image_bytes : bytes):
    image = types.Part.from_bytes(
        data=image_bytes, mime_type="image/jpeg"
    )

    model_result = await detect_trash(image_bytes)
    if not model_result:
        return AiResponse(
            trash_detected=False,
            is_fake=None,
            is_indoor=None,
            trash_type=None,
            volume_estimate=None,
            ai_analysis=None,
            severity_score=None
        )

    prompt = """
                You are an AI waste detection analyst. Analyze this image.

                Analyze the image and respond ONLY with a valid JSON object in this exact format, no other text:
                {
                    "trash_detected": true or false,
                    "is_fake": true or false,
                    "is_indoor": true or false,
                    "trash_type": one of ["PLASTIC", "ORGANIC", "HAZARDOUS", "CONSTRUCTION", "MIXED", "OTHER"],
                    "volume_estimate": one of ["SMALL", "MEDIUM", "LARGE"],
                    "ai_analysis": "brief description of what you see, confidence level, and any concerns",
                    "severity_score": a number between 1.0 and 10.0
                }

                Rules:
                - trash_detected: true only if there is clearly visible waste/trash in the image
                - is_fake: true if the image appears to be indoor, a stock photo, screenshot, or not a real outdoor waste scenario
                - is_indoor: true if the location appears to be inside a building
                - trash_type: best matching category for the primary waste visible
                - volume_estimate: SMALL (less than 1 sqm), MEDIUM (1-5 sqm), LARGE (more than 5 sqm)
                - severity_score: 1 is minor litter, 10 is severe hazardous dump. Consider volume, type, and location context
                - ai_analysis: keep it under 30 words
            """

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt, image]
    )

    text = response.text.strip().replace("```json", "").replace("```", "")
    data = json.loads(text)
    return AiResponse(**data)