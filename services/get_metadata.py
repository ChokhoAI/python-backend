from PIL.ExifTags import Base , GPS , GPSTAGS
from PIL import Image
from fastapi import UploadFile
import cv2

def get_metadata(image_path):
    image = Image.open(image_path)

    exif_data = image.getexif()

    metadata = {}

    # if exif_data:
    #     for tag, value in exif_data.items():
            

    return 0


get_metadata("../images/waste2.jpg")