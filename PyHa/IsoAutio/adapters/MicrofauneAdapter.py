from typing import Callable

import pandas as pd

from ...microfaune_package.microfaune.detection import RNNDetector

from ..BaseAdapter import BaseAdapter
from ..Result import Result
from ..Audio import Audio


class MicrofauneAdapter(BaseAdapter):
    def generate(self, callback: Callable[[Result], pd.DataFrame]) -> pd.Dataframe:
        detector = (
            RNNDetector()
            if self.config.weight_path is None
            else RNNDetector(self.config.weight_path)
        )

        def create_entry(audio: Audio) -> pd.DataFrame:
            microfaune_features = detector.compute_features([audio.signal])
            _, local_scores = detector.predict(microfaune_features)
            return callback(Result(self.config, audio, local_scores))

        return self.__generate_annotations(create_entry)
