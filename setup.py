#!/usr/bin/env python
"""Setup script for the zerospeech2020 Python package"""

import codecs
import setuptools
import zerospeech2021

setuptools.setup(
    # general description
    name='zerospeech2021',
    description="Evaluation and validation tools for ZeroSpeech2021",
    version=zerospeech2021.__version__,

    # python package dependencies
    install_requires=['numpy', 'pyyaml', 'joblib', 'pandas'],
    setup_requires=[],

    # include Python code and any file in zerospeech2021/share
    packages=setuptools.find_packages(),
    package_data={'zerospeech2021': ['share/*']},
    zip_safe=True,

    # the command-line scripts to export
    entry_points={'console_scripts': [
        'zerospeech2021-validate = zerospeech2021.validation.main:main',
        'zerospeech2021-evaluate = zerospeech2021.evaluation.main:main']},

    # metadata
    author='CoML team',
    author_email='zerospeech2021@gmail.com',
    license='GPL3',
    url='https://zerospeech.com/2021',
    long_description=codecs.open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    python_requires='>=3.7',
)
