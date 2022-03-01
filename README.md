# ZeroSpeech Challenge 2021 Python package


This repository bundles all the scripts required to evaluate and validate a
submission to the [ZeroSpeech Challenge 2021](https://zerospeech.com/2021).

## Installation

* First clone this repository

        git clone https://github.com/bootphon/zerospeech2021.git
        cd zerospeech2021

* Setup a conda environment:

        conda env create -f environment.yml

* Activate the created environment:

        conda activate zerospeech2021

* Install the package:

        python setup.py install

## Usage

The `zerospeech2021` package provides 2 command-line tools:

* `zerospeech2021-validate` which validates a submission, ensuring all the
  required files are here and correctly formatted.

* `zerospeech2021-evaluate` which evaluates a submission (supposed valid). Only
  the development datasets are evaluated. The test datasets can only be
  evaluated by doing an official submission to the challenge.

* `zerospeech2021-leaderboard` which allows generation of leaderboard entries from scores.

* `zerospeech2021-upload` utility to allow upload submission to zerospeech.com.

Each tool comes with a `--help` option describing the possible arguments (e.g.
`zerospeech2021-validate --help`).
