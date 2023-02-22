from Multitask_Roberta import RobertaForMultipleTask
from transformers import XLMRobertaConfig

class XLMRobertaForMultipleTask(RobertaForMultipleTask):
    """
    This class overrides [`RobertaForMultipleTask`]. Please check the superclass for the appropriate documentation alongside
    usage examples.
    """

    config_class = XLMRobertaConfig