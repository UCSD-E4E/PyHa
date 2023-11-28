from typing import Union


def check_verbose(message: str, verbose: Union[bool, dict]):
    """Adds the ability to toggle on/off error messages and warnings


    Args:
        message (str): Error message
        verbose (Union[bool, dict]): Log the error message (or not)
    """

    if verbose == True or (isinstance(verbose, dict) and verbose["verbose"]):
        print(message)
