import base64
import ipaddress
import logging
import socket
from io import BytesIO
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import ExifTags, Image

from .gs import get_gcs_from_http_url, read_file_bytes

DOWNLOAD_TIMEOUT_SECONDS = 30


class UnsafeUrlError(ValueError):
    pass


def validate_url(http_url):
    # Reject URLs that could be used for SSRF: the request runs server-side
    # with the service account's network access, so only allow http(s) URLs
    # that resolve to public IPs (blocks the GCE metadata server, localhost,
    # and private/internal ranges).
    parsed = urlparse(http_url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeUrlError(f"unsupported URL scheme: {parsed.scheme}")
    if not parsed.hostname:
        raise UnsafeUrlError("URL has no hostname")
    try:
        addr_infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        raise UnsafeUrlError(f"could not resolve host: {parsed.hostname}") from e
    for addr_info in addr_infos:
        ip = ipaddress.ip_address(addr_info[4][0])
        if not ip.is_global:
            raise UnsafeUrlError(
                f"URL host {parsed.hostname} resolves to non-public address {ip}"
            )


def read_image_from_url(http_url, http_as_gcs=False) -> Image.Image:
    # Extract GSC URI from http link and download the file directly.
    # It will increase reliability, as it will use GCP driver to fetch data
    # If URL is not GCS HTTP URL, download it through HTTP
    if http_as_gcs:
        gs_uri = get_gcs_from_http_url(http_url)
        if gs_uri:
            return read_image_from_gs(gs_uri)

    validate_url(http_url)
    response = requests.get(http_url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
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
    orientation = get_orientation(image)
    if orientation:
        logging.info(f"Rotating image for orientation {orientation}")
        image = rotate_image(image, orientation)

    return clean_exif(image)


def image_to_base64(image):
    bytes_content = image_to_bytes(image)
    return base64.b64encode(bytes_content).decode("utf-8")


def mask_to_base64(mask: np.ndarray) -> str:
    buffered = BytesIO()
    Image.fromarray(mask).save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def array_to_base64_npy(array: np.ndarray) -> str:
    """Encode an arbitrary-dtype array as base64 .npy bytes (lossless).

    Decode with np.load(BytesIO(base64.b64decode(data))).
    """
    buffered = BytesIO()
    np.save(buffered, array)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def image_to_bytes(image):
    buffered = BytesIO()
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return buffered.getvalue()


def base64_to_image(base64_data: str):
    return Image.open(BytesIO(base64.b64decode(base64_data)))


def image_to_array(image: Image.Image):
    return np.array(image.convert("RGB"))


def pil_to_opencv(image: Image.Image):
    return cv2.cvtColor(image_to_array(image), cv2.COLOR_RGB2BGR)


def opencv_to_pil(image: np.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)
