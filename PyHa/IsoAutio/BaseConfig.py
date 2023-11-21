from typing import Optional
from Result import ThresholdMethod

import logging


class BaseConfig:
    def __init__(
        self,
        audio_dir: str,
        manual_id: str = "bird",
        weight_path: str = None,
        normalized_sample_rate: int = 44100,
        normalize_local_scores: bool = False,
        threshold_method: Optional[ThresholdMethod] = "median",
        threshold_const: Optional[float] = None,
        threshold_min: Optional[float] = 0,
        logging_level: int = logging.DEBUG,
    ):
        self.audio_dir = audio_dir
        self.manual_id = manual_id
        self.weight_path = weight_path
        self.normalized_sample_rate = normalized_sample_rate
        self.normalize_local_scores = normalize_local_scores

        # result config
        self.threshold_method = threshold_method
        self.threshold_const = threshold_const
        self.threshold_min = threshold_min

        if not normalized_sample_rate > 0:
            logging.error("normalized_sample_rate should be greater than 0")

        # setup logging
        logging.basicConfig()
        logging.getLogger().setLevel(logging_level)
