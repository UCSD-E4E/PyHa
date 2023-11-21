from typing import Callable

import pandas as pd
import torch

from ...tweetynet_package.tweetynet.TweetyNetModel import TweetyNetModel
from ...tweetynet_package.tweetynet.Load_data_functions import (
    compute_features,
    predictions_to_kaleidoscope,
)

from ..BaseAdapter import BaseAdapter
from ..Result import Result
from ..Audio import Audio


class TweetyNetAdapter(BaseAdapter):
    def generate(
        self, callback: Callable[[Result], pd.DataFrame], original: bool = True
    ) -> pd.Dataframe:
        detector = TweetyNetModel(2, (1, 86, 86), 86, torch.device("cpu"))

        def create_entry(audio: Audio) -> pd.DataFrame:
            features = compute_features([audio.signal])
            predictions, local_scores = detector.predict(
                features,
                model_weights=self.config.weight_path,
                norm=self.config.normalize_local_scores,
            )

            if original:
                return predictions_to_kaleidoscope(
                    predictions,
                    audio.signal,
                    self.config.audio_dir,
                    audio.filename,
                    self.config.manual_id,
                    audio.sample_rate,
                )

            return callback(Result(self.config, audio, local_scores))

        return self.__generate_annotations(create_entry)
