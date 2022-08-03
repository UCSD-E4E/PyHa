from setuptools import setup, find_packages

packages = find_packages()

setup(
    name='microfaune',
    version='0.0',
    packages=packages,
    long_description='Package used for the microfaune project',
    package_data={'': ['data/*.h5']}
)
