import base64
import logging
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ExifTags
import requests

from .gs import get_gcs_from_http_url, read_file_bytes


def read_image_from_url(http_url, http_as_gsc=False) -> Image.Image:
    # Extract GSC URI from http link and download the file directly.
    # It will increase reliability, as it will use GCP driver to fetch data
    # If URL is not GCS HTTP URL, download it through HTTP
    if http_as_gsc:
        gs_uri = get_gcs_from_http_url(http_url)
        if gs_uri:
            return read_image_from_gs(gs_uri)

    response = requests.get(http_url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return image


def read_image_from_gs(gs_url) -> Image.Image:
    bytes_content = read_file_bytes(gs_url)
    image = Image.open(BytesIO(bytes_content))
    return image


def get_orientation(image):
    try:
        exif = image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == "Orientation":
                    return value
    except (AttributeError, KeyError, IndexError):
        return None


def rotate_image(image, orientation):
    if orientation == 3:
        image = image.rotate(180, expand=True)
    elif orientation == 6:
        image = image.rotate(270, expand=True)
    elif orientation == 8:
        image = image.rotate(90, expand=True)
    return image


def clean_exif(image):
    if hasattr(image, "_getexif"):
        image.info.pop("exif", None)
        if hasattr(image, "_exif"):
            image._exif = None
    return image


def fix_rotation(image):
    """
    Encode input image to base64, ensure it's rotated without EXIF tags
    """
    orientation = get_orientation(image)
    if orientation:
        logging.info(f"Rotating image for orientation {orientation}")
        image = rotate_image(image, orientation)

    return clean_exif(image)


def image_to_base64(image):
    bytes_content = image_to_bytes(image)
    return base64.b64encode(bytes_content).decode("utf-8")


def image_to_bytes(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return buffered.getvalue()


def pil_to_opencv(image: Image.Image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def opencv_to_pil(image: np.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)
