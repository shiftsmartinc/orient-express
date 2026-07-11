"""Helpers shared by the kserve inference servers in docker-image-onnx/ and docker-xgboost-scikit-learn/.

kserve itself is deliberately not imported here — it is a `server`
dependency-group package that only exists inside the Docker images.
"""

import inspect
import json
import logging
import threading

import requests
import requests.adapters

from .utils.image_processor import (
    base64_to_image,
    read_image_from_gs,
    read_image_from_url,
)
from .utils.retry import retry

_http_session: requests.Session | None = None
_http_session_lock = threading.Lock()


def get_http_session() -> requests.Session:
    """Shared session so per-image downloads reuse connections (keep-alive)
    instead of paying DNS + TCP + TLS setup per request."""
    global _http_session
    if _http_session is None:
        with _http_session_lock:
            if _http_session is None:
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=8, pool_maxsize=32
                )
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                _http_session = session
    return _http_session


# Defaults the server supplies when a predictor's predict() requires a
# parameter the request didn't provide.
SERVER_PARAM_DEFAULTS = {"confidence": 0.5}

# Request parameters that steer the server itself and must not be forwarded
# to predict().
RESERVED_PARAMETERS = {"debug_image"}


def decode_input(input_data):
    logging.info(f"PayloadType: {type(input_data)}")
    if isinstance(input_data, (bytes, str)):
        return json.loads(input_data)
    elif isinstance(input_data, dict):
        return input_data
    else:
        raise Exception(f"unsupported payload type {type(input_data)}")


@retry(retries=3)
def download_image(image_address):
    if image_address.startswith("http"):
        logging.info(f"image source: http url {image_address}")
        return read_image_from_url(image_address, session=get_http_session())
    elif image_address.startswith("gs://"):
        logging.info(f"image source: gcs uri {image_address}")
        return read_image_from_gs(image_address)
    elif image_address.startswith("data:"):
        # data URI format: data:image/png;base64,iVBORw0KGgo...
        base64_data = image_address.split(",", 1)[1]
        logging.info(f"image source: base64 data uri ({len(base64_data)} base64 chars)")
        return base64_to_image(base64_data)
    else:
        logging.info(f"image source: raw base64 ({len(image_address)} base64 chars)")
        return base64_to_image(image_address)


def build_predict_kwargs(predict_fn, parameters: dict) -> dict:
    """Map request `parameters` onto the predictor's own predict() signature.

    Any parameter the signature declares is forwarded when present in the
    request; required parameters missing from the request fall back to
    SERVER_PARAM_DEFAULTS. Unknown request keys and RESERVED_PARAMETERS are
    ignored, so the server needs no per-model-type dispatch.
    """
    signature = inspect.signature(predict_fn)
    kwargs = {}
    for name, param in signature.parameters.items():
        if name in ("self", "images"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in RESERVED_PARAMETERS:
            continue
        if name in parameters:
            kwargs[name] = parameters[name]
        elif param.default is inspect.Parameter.empty and name in SERVER_PARAM_DEFAULTS:
            kwargs[name] = SERVER_PARAM_DEFAULTS[name]
    return kwargs
