REGISTRY = {}


from .rnn_agent import RNNAgent
from .rnn_agent_share import RNNAgentShare
from .rnn_agent_rmix import RmixAgent
from .rnn_agent_rmix_share import RmixAgentShare

from .iqn_rnn_agent import IQNRNNAgent
from .iqn_rnn_agent_share import IQNRNNAgentShare



REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_share"] = RNNAgentShare

REGISTRY["iqn_rnn"] = IQNRNNAgent
REGISTRY["iqn_rnn_share"] = IQNRNNAgentShare


REGISTRY["rnn_agent_rmix"] = RmixAgent
REGISTRY["rnn_agent_rmix_share"] = RmixAgentShare

