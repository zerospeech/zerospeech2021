""" Phonetic task zerospeech 2021 """
import collections
from dataclasses import dataclass
from pathlib import Path
from itertools import chain
from typing import Optional
from enum import Enum

import numpy as np
import yaml

from zerospeech2021 import exception
from zerospeech2021.phonetic_eval import eval_ABX

LIBRISPEECH_SETS = {
    'dev': ['dev-clean', 'dev-other'],
    'test': ['test-clean', 'test-other']}


ABXFileTypes = Enum('ABXFileTypes',
                    '.pt .npy .txt .wav .flac .mp3')
ABXMode = Enum('ABXMode', 'all within across')

ABXDistanceMode = Enum('ABXDistanceMode',
                       'euclidian cosine kl kl_symmetric')


@dataclass
class AbxArguments:
    """ List of arguments to provide to abx in phonetic_eval.abx"""
    # path to input data
    path_data: str
    # path to item file
    path_item_file: str
    # Path to a CPC checkpoint
    path_checkpoint: Optional[str] = None
    # size of a single feature
    feature_size: Optional[float] = float(0.1)
    # Use the GPU to compute distances
    cuda: bool = True
    # extension (of input files ?)
    file_extension: ABXFileTypes = '.txt'
    # Choose the mode of the ABX score to compute
    mode: ABXMode = 'all'
    # Choose the kind of distance to use to compute
    distance_mode: ABXDistanceMode = 'cosine'
    # Max size of a group while computing the ABX score
    max_size_group: int = 10
    # When computing the ABX across score, maximum
    # number of speaker X to sample per couple A,B.
    max_x_across: int = 5
    # location to output the results
    out: Optional[str] = None


def load_meta_args(features_location: Path):
    with (features_location / 'meta.yaml').open() as fp:
        meta = yaml.safe_load(fp)
    try:
        metric = meta['parameters']['phonetic']['metric']
    except KeyError:
        raise ValueError("The metric must be specified in the meta.yaml phonetic section")

    try:
        features_size = float(meta['parameters']['phonetic']['features_size'])

        return dict(features_size=features_size, metric=metric)
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
    """ Verifies that a feature file is a 2D numpy array of floats

    :raises exception.FileFormatError if the types are not correct
    :raises ValueError if the file is not a valid numpy array

    :return: the number of columns in the array
    """
    try:
        array = np.loadtxt(str(feature_path))
    except Exception:
        raise exception.FileFormatError(
            feature_path, 'not a valid numpy array')

    if array.dtype != np.dtype('float'):
        raise exception.FileFormatError(
            feature_path, "array loaded is not dtype = float")

    if array.ndim != 2:
        raise exception.FileFormatError(
            feature_path, 'not a 2D array')

    return array.shape[1]


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
    ncols = None
    valid_entries = []
    for file in input_files:
        pure_path = file.relative_to(dataset_directory)
        txt_file = submission_directory / pure_path
        txt_file = txt_file.with_suffix('.txt')
        if not txt_file.is_file():
            raise exception.EntryMissingError(source=file, expected=txt_file)

        current_ncols = verify_feature_file(txt_file)
        if ncols and current_ncols != ncols:
            raise exception.FileFormatError(
                txt_file, f'expected {ncols} columns but get {current_ncols}')
        ncols = current_ncols

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

    input_files = get_input_files(dataset, _set, "wav")
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


def evaluate(features_location: Path, dataset: Path, output_dir: Path, kind):
    meta_values = load_meta_args(features_location.parents[0])
    metric = meta_values.get("metric", 'cosine')
    feature_size = meta_values.get("feature_size", 0.01)

    for _set in LIBRISPEECH_SETS[kind]:
        arg_obj = AbxArguments(
            path_data=str(features_location / _set), path_item_file=f'{(dataset / _set / f"{_set}.item")}',
            distance_mode=f"{metric}", feature_size=feature_size,
            out=f"{output_dir}"
        )
        eval_ABX.main(arg_obj=arg_obj)
