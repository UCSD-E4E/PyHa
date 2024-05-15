'''Test Fixtures
'''
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

import pytest
from nas_unzip.nas import nas_unzip


@pytest.fixture(name='creds', scope='session')
def create_creds() -> Dict[str, str]:
    """Obtains the credentials

    Returns:
        Dict[str, str]: Username and password dictionary
    """
    if Path('credentials.json').is_file():
        with open('credentials.json', 'r', encoding='ascii') as handle:
            return json.load(handle)
    else:
        return json.loads(os.environ['NAS_CREDS'])
        # value = os.environ['NAS_CREDS'].splitlines()
        # assert len(value) == 2
        # return {
        #     'username': value[0],
        #     'password': value[1]
        # }


@pytest.fixture(name='reference_data', scope='session')
def create_reference_data(creds) -> Path:
    """Creates reference data

    Args:
        creds (Dict[str, str]): NAS Credentials

    Returns:
        Path: Path to data

    Yields:
        Iterator[Path]: Path to data
    """
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir).resolve()
        if not Path('TEST').is_dir():
            nas_unzip(
                network_path='smb://e4e-nas.ucsd.edu:6021/temp/github_actions/pyha/pyha_test.zip',
                output_path=path,
                username=creds['username'],
                password=creds['password']
            )
            yield path.joinpath('TEST')
        else:
            #shutil.copytree(Path('TEST'), path, dirs_exist_ok=True)
            yield path.joinpath('TEST')
