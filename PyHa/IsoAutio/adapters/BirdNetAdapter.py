import pandas as pd

from ...birdnet_lite.analyze import analyze

from ..BaseAdapter import BaseAdapter


class BirdNetAdapter(BaseAdapter):
    def generate(
        self,
        lat: float = -1,
        lon: float = -1,
        week: int = -1,
        overlap: float = 0.0,
        sensitivity: float = 1.0,
        min_conf: float = 0.1,
        custom_list: str = "",
        filetype: str = "wav",
        num_predictions: int = 10,
        # @audit these options should be condensed
        write_to_csv: bool = False,
        output_path: str = None,
    ) -> pd.Dataframe:
        """Function that generates the bird labels for an audio file or across a folder using the BirdNet-Lite model

        Args:
            lat (float, optional): Recording location latitude. Defaults to -1.
            lon (float, optional): Recording location longitude. Defaults to -1.
            week (int, optional): Week of the year when the recording was made. Values in [1, 48]. Defaults to -1.
            overlap (float, optional): Overlap in seconds between extracted spectrograms. Values in [0.5, 1.5]. Defaults to 0.0.
            sensitivity (float, optional): Detection sensitivity. Higher values result in higher sensitivity. Values in [0.5, 1.5]. Defaults to 1.0.
            min_conf (float, optional): Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.
            custom_list (str, optional): Path to text file containing a list of species. Defaults to "".
            filetype (str, optional): Filetype of soundscape recordings. Defaults to "wav".
            num_predictions (int, optional): Defines maximum number of written predictions in a given 3s segment. Defaults to 10.
            write_to_csv (bool, optional): Set whether or not to write output to CSV. Defaults to False.
            output_path (str, optional): Path to output folder. By default results are written into the input folder. Defaults to None.

        Returns:
            pd.Dataframe: Dataframe of automated labels for the audio clip(s) in audio_dir.
        """

        return analyze(
            audio_path=self.config.audio_dir,
            lat=lat,
            lon=lon,
            week=week,
            overlap=overlap,
            sensitivity=sensitivity,
            min_conf=min_conf,
            custom_list=custom_list,
            filetype=filetype,
            num_predictions=num_predictions,
            write_to_csv=write_to_csv,
            output_path=output_path,
        )
