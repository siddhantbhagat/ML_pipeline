from setuptools import setup,find_packages
from typing import List


def get_requirements(req_filepath:str)->List[str]:
    """
    Gathers requirements form requirements.txt.

    Args:
        req_filepath (str): file path of the requirements.txt.

    Returns:
        List[str]: List of packages that needs to be installed.
    """
    requirements = []
    with open(req_filepath,'r') as txtfile:
        requirements = txtfile.readlines()
        requirements = [req.replace('\n','') for req in requirements]

    if '-e .' in requirements:
        requirements.remove('-e .')

    return requirements

setup(
    name='ML_Project',
    version='0.0.1',
    author='Siddhant Bhagat',
    author_email='siddhantbhgt@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)