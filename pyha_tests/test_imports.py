'''Tests PyHa environment
'''
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

from nas_unzip.nas import nas_unzip
import torch

from PyHa.IsoAutio import generate_automated_labels


def create_creds() -> Dict[str, str]:
    """Obtains the credentials

    Returns:
        Dict[str, str]: Username and password dictionary
    """
    if Path('credentials.json').is_file():
        with open('credentials.json', 'r', encoding='ascii') as handle:
            return json.load(handle)
    else:
        value = os.environ['NAS_CREDS'].splitlines()
        assert len(value) == 2
        return {
            'username': value[0],
            'password': value[1]
        }


def run_pyha(creds):
    """Tests PyHa
    """
    with TemporaryDirectory() as path:
        nas_unzip(
            network_path='smb://e4e-nas.ucsd.edu:6021/temp/github_actions/pyha/pyha_test.zip',
            output_path=Path(path),
            username=creds['username'],
            password=creds['password']
        )
        isolation_parameters = {
            "model": "tweetynet",
            "tweety_output": True,
            "technique": "steinberg",
            "threshold_type": "median",
            "threshold_const": 2.0,
            "threshold_min": 0.0,
            "window_size": 2.0,
            "chunk_size": 5.0,
            "verbose": True
        }
        # generate_automated_labels(path, isolation_parameters)
        # isolation_parameters = {
        #     "model": "birdnet",
        #     "output_path": "outputs",
        #     "lat": 35.4244,
        #     "lon": -120.7463,
        #     "week": 18,
        #     "min_conf": 0.1,
        #     "filetype": "wav",
        #     "num_predictions": 1,
        #     "write_to_csv": False,
        # }
        # generate_automated_labels(path, isolation_parameters)
        # isolation_parameters = {
        #     "model":          "microfaune",
        #     "technique":       "steinberg",
        #     "threshold_type":  "median",
        #     "threshold_const": 2.0,
        #     "threshold_min":   0.0,
        #     "window_size":     2.0,
        #     "chunk_size":      5.0,
        #     "verbose":     True
        # }
        # generate_automated_labels(path, isolation_parameters)

def main():
    creds = create_creds()
    run_pyha(creds)

if __name__ == '__main__':
    main()