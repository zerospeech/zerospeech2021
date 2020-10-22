# ZeroSpeech Challenge 2021 Python package


This repository bundles all the scripts required to evaluate and validate a
submission to the [ZeroSpeech Challenge 2021](https://zerospeech.com/2021).

## Installation

TODO

## Usage

The `zerospeech2020` package provides 2 command-line tools:

* `zerospeech2020-validate` which validates a submission, ensuring all the
  required files are here and correctly formatted.

* `zerospeech2020-evaluate` which evaluates a submission (supposed valid). Only
  the development datasets are evaluated. The test datasets can only be
  evaluated by doing an official submission to the challenge.

Each tool comes with a `--help` option describing the possible arguments (e.g.
`zerospeech2021-validate --help`).
