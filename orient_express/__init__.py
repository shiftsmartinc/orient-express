"""Orient Express — model deployment to Vertex AI.

Top-level names are imported lazily (PEP 562): `import orient_express` costs
milliseconds, and heavy dependencies (google-cloud-aiplatform, onnxruntime,
cv2) load only when the first attribute that needs them is touched.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model_wrapper import JoblibSimpleLoader, ModelExpress

_EXPORTS = {
    "JoblibSimpleLoader": ".model_wrapper",
    "ModelExpress": ".model_wrapper",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
