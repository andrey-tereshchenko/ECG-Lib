from pathlib import Path
from typing import List
from setuptools import setup


def parse_requirements() -> List[str]:
    path = Path() / 'requirements.txt'
    with open(str(path), 'r') as file:
        data = file.read()
    lines = data.split('\n')
    return [line.strip() for line in lines if line.strip()]


setup(name='ecg_lib',
      version='0.1',
      url='https://github.com/andrey-tereshchenko/ECG-Lib',
      license='MIT',
      author='Andrii Tereshchenko',
      author_email='avtogol1998@gmail.com',
      description='Library for build word2vec model based on ECG signal',
      install_requires=parse_requirements(),
      zip_safe=False)
