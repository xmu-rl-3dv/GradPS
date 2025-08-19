REGISTRY = {}

from .basic_controller import BasicMAC
from .rmix_controller import RmixMAC
from .dmix_controller import DmixMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["rmix_mac"] = RmixMAC
REGISTRY["dmix_mac"] = DmixMAC
