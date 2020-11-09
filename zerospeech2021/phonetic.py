""" Phonetic task zerospeech 2021 """
import collections
from pathlib import Path
from itertools import chain

import numpy as np
import yaml

from zerospeech2021 import exception
from zerospeech2021.phonetic_eval import eval_ABX

LIBRISPEECH_SETS = {
    'dev': ['dev-clean', 'dev-other'],
    'test': ['test-clean', 'test-other']}


def load_meta_args(features_location: Path):
    with (features_location / 'meta.yaml').open() as fp:
        meta = yaml.safe_load(fp)
    try:
        metric = meta['parameters']['phonetic']['metric']
    except KeyError:
        metric = "cosine"

    try:
        file_extension = meta['parameters']['file_type']
    except KeyError:
        file_extension = 'wav'

    try:
        features_size = meta['parameters']['phonetic']['features_size']

        return features_size, metric, file_extension
    except KeyError:
        raise ValueError("feature size must be defined in the meta.yaml")


def get_input_files(dataset_directory, _set, file_type):
    """ Returns a list of all the files in a set """
    res = []
    for s in LIBRISPEECH_SETS[_set]:
        res.append((dataset_directory / s).rglob(f"*.{file_type}"))
    return list(chain(*res))


def get_submitted_files(submission_directory, _set):
    """ Returns a list of all the files in a set """
    res = []
    for s in LIBRISPEECH_SETS[_set]:
        res.append((submission_directory / s).rglob("*.txt"))
    return list(chain(*res))


def verify_feature_file(feature_path: Path):
    """ Verifies that a feature file is a parsable numpy array of floats
        :raises exception.FileFormatError if the types are not correct
        :raises ValueError if the file is not parsable by numpy
    """
    array = np.loadtxt(str(feature_path))
    if not array.dtype == np.dtype('float'):
        raise exception.FileFormatError(
            feature_path, "array loaded is not dtype = float")


def check_entries(
        input_files, submission_directory, dataset_directory, _set):
    """ Checks all entries from the input dataset to see if they match they
        exist in the submitted set.
    :param input_files: list of input files (from dataset)
    :param submission_directory: location of submitted files
    :param dataset_directory: location of dataset
    :param _set: type of subset (test, dev)
    :return: a list of valid entries
    :raises exception.EntryMissingError if an entry is not present
    """
    valid_entries = []
    for file in input_files:
        pure_path = file.relative_to(dataset_directory)
        txt_file = submission_directory / pure_path
        txt_file = txt_file.with_suffix('.txt')
        if not txt_file.is_file():
            raise exception.EntryMissingError(source=file, expected=txt_file)
        verify_feature_file(txt_file)
        valid_entries.append(txt_file)
    return valid_entries


def validate(submission, dataset, _set):
    """  Validate a subset of the submissions for the phonetic task

    :param submission_directory: location of submissions
    :param dataset_directory: location of data
    :param file_type: entry files (wav | flac)
    :param _set: subset type (dev | test)
    """

    if _set not in LIBRISPEECH_SETS.keys():
        raise ValueError(f'kind must be "dev" or "test", it is {_set}')

    input_files = get_input_files(dataset, _set, "*.wav")
    if not input_files:
        raise exception.ValidationError(
            f'found no wav files in {dataset}')

    submitted_files = get_submitted_files(submission, _set)
    if not input_files:
        raise exception.ValidationError(
            f'found no .txt files in {submission}')

    # ensure that there are no duplicates
    duplicates = [
        f for f, n in collections.Counter(submitted_files).items() if n > 1
    ]
    if duplicates:
        raise exception.MismatchError('duplicates found', [], duplicates)

    # check that necessary files are present and valid
    valid_entries = check_entries(
        input_files, submission, dataset, _set)

    if collections.Counter(submitted_files) != collections.Counter(valid_entries):
        raise exception.MismatchError(
            'mismatch in filenames', valid_entries, submitted_files)


def evaluate(features_location: Path, abx_data: Path, output_dir: Path, _set):
    metric, feature_size, file_extension = load_meta_args(features_location)

    # todo maybe add some more parameters
    args = [
        f"{features_location}",
        "<item file>",
        f"--file_extension {file_extension}",
        f"--out {output_dir}",
        f"--feature_size {feature_size}",
        f"--distance_mode {metric}",
        "--cuda"
    ]

    for s in LIBRISPEECH_SETS[_set]:
        args[1] = (abx_data / f"{s}.item")
        eval_ABX.main(args)
