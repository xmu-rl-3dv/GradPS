from .q_learner import QLearner
from .q_share_learner import QShareLearner

from .dmaq_qatten_learner import DMAQ_qattenLearner
from .dmaq_qatten_share_learner import DMAQ_qattenShareLearner
from .rmix_learner import RMIXLearner
from .rmix_share_learner import RMIXShareLearner
from .iqn_learner import IQNLearner
from .iqn_share_learner import IQNShareLearner


REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["q_share_learner"] = QShareLearner

REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["dmaq_qatten_share_learner"] = DMAQ_qattenShareLearner

REGISTRY["rmix_learner"] = RMIXLearner
REGISTRY["rmix_share_learner"] = RMIXShareLearner

REGISTRY["iqn_learner"] = IQNLearner
REGISTRY["iqn_share_learner"] = IQNShareLearner
