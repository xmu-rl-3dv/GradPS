from .run import run as default_run
from .dop_run import run as dop_run

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["dop_run"] = dop_run