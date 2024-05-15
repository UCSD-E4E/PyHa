'''Tests PyHa environment
'''
from pathlib import Path
import importlib
import os

from PyHa.IsoAutio import generate_automated_labels


def test_reference_data(reference_data: Path):
    """Tests that the reference data is sane

    Args:
        reference_data (Path): Path to reference data
    """
    print(reference_data)
    print(Path("/tmp").exists())
    print(os.listdir("/tmp"))
    assert reference_data.exists()
    assert reference_data.joinpath("TEST").exists()
    for i in range(11):
        assert reference_data.joinpath(f'ScreamingPiha{i + 1}.wav').exists()

def test_resampy_installed():
    try:
        importlib.import_module('resampy')
        assert True  # resampy is installed
    except ImportError:
        assert False, "resampy is not installed"
        
def test_pyha(reference_data: Path):
    """Tests PyHa
    """
    
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
    df = generate_automated_labels(reference_data.as_posix() + '/', isolation_parameters)
    assert df.empty != True

    isolation_parameters = {
        "model": "birdnet",
        "output_path": "outputs",
        "lat": 35.4244,
        "lon": -120.7463,
        "week": 18,
        "min_conf": 0.1,
        "filetype": "wav",
        "num_predictions": 1,
        "write_to_csv": False,
    }
    df = generate_automated_labels(reference_data.as_posix() + '/', isolation_parameters)
    assert df.empty != True

    isolation_parameters = {
        "model":          "microfaune",
        "technique":       "steinberg",
        "threshold_type":  "median",
        "threshold_const": 2.0,
        "threshold_min":   0.0,
        "window_size":     2.0,
        "chunk_size":      5.0,
        "verbose":     True
    }
    df = generate_automated_labels(reference_data.as_posix() + '/', isolation_parameters)
    assert df.empty != True 