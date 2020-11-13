"""Semantic part of the ZR2021 (validation and evaluation)"""

import functools
import pathlib

import numpy as np
import pandas
import scipy.spatial
import joblib

from zerospeech2021.exception import (
    MismatchError, FileFormatError, ValidationError, EntryMissingError)


def _validate_file(source_file, submission):
    """Ensure a file has the correct format

    Verifies that a feature file is a 2D numpy array of floats.

    :param source_file: input file
    :param submission: location of submitted files
    :return: a pair (error, ncols)

    """
    try:
        target_file = submission / (source_file + '.txt')
        if not target_file.is_file():
            raise EntryMissingError(
                source=source_file, expected=target_file)

        try:
            array = np.loadtxt(str(target_file))
        except Exception:
            raise FileFormatError(target_file, 'not a valid numpy array')

        if array.dtype != np.dtype('float'):
            raise FileFormatError(target_file, "not a float array")

        if array.ndim != 2:
            raise FileFormatError(target_file, 'not a 2D array')

    except ValidationError as error:
        return str(error), None

    return None, array.shape[1]


def validate(submission, dataset, kind, subset, njobs=1):
    """Raises a ValidationError if the `submission` is not valid

    The submission folder must include <filename>.txt files, each file
    containing a matrix of floats. Each <filename>.wav file in the dataset must
    have its <filename>.txt equivalent in the submission directory.

    Parameters
    ----------
    submisison: path
        The submisison directory to validate.
    dataset: path
        The root path of the ZR2021 dataset.
    kind: str
        Must be 'dev' or 'test'.
    subset: str
        Must be 'synthetic' or 'librispeech'
    njobs : int
        Number of parallel processes to use

    Raises
    ------
    ValueError
        If `kind` is not 'dev' or 'test', if `submisison` or `dataset` are not
        an existing directory.
    ValidationError
        If one line of the submission file is not valid or if the submitted
        filenames does not fit the required ones.

    """
    if kind not in ('dev', 'test'):
        raise ValueError(
            f'kind must be "dev" or "test", it is {kind}')

    if subset not in ('librispeech', 'synthetic'):
        raise ValueError(
            f'subset must be "librispeech" or "synthetic", it is {subset}')

    submission = pathlib.Path(submission) / kind / subset
    if not submission.is_dir():
        raise ValueError(
            f'{kind} submission directory not found: {submission}')

    dataset = pathlib.Path(dataset) / f'semantic/{kind}/{subset}'
    if not dataset.is_dir():
        raise ValueError(f'dataset not found: {dataset}')

    # retrieve the required filenames that must be present in the submission
    required = set(f.stem for f in dataset.glob('*.wav'))
    if not required:
        raise ValidationError(f'{dataset} contains no .wav files')

    # retrieve the submitted files
    submitted = set(submission.glob('*'))
    if not submitted:
        raise ValidationError(f'{submission} contains no files')

    # ensure we have only .txt files in submission
    no_txt_files = [str(f) for f in submitted if f.suffix != '.txt']
    if no_txt_files:
        raise MismatchError('extra files found', [], no_txt_files)

    # ensure each required file is present in the submission
    submitted = set(f.stem for f in submitted)
    if submitted != required:
        raise MismatchError('files mismatch', required, submitted)

    # ensure each submitted file has a correct format ad the number of columns
    # is constant across files
    errors, ncols = zip(*joblib.Parallel(n_jobs=njobs)(
        joblib.delayed(_validate_file)(f, submission) for f in submitted))

    # ensure there are no detected errors
    errors = [e for e in errors if e]
    if errors:
        for e in errors[:10]:
            print(f'ERROR: {e}')
        if len(errors) > 10:
            print('ERROR: ... and {len(errors - 10)} more!')
        raise ValidationError(f'error detected in phonetic {kind}')

    # ensure all submitted files have the same number of columns
    if len(set(ncols)) != 1:
        raise ValidationError(
            f'all files must have the same number of columns '
            f'but have: {set(ncols)}')


class EvaluationHelper:
    """This is a helper class for semantic evaluation

    Given a pair of words, this class (using the distance() method) retrieves
    the corresponding word tokens, load them, apply pooling and compute the
    mean distance between each token of the two words.

    """
    def __init__(self, submission_dir, gold_file, metric, pooling):
        self._folder = pathlib.Path(submission_dir)
        if not self._folder.is_dir():
            raise ValueError(f'not a directory: {self._folder}')

        self._gold = pandas.read_csv(gold_file, header=0)
        self._metric = metric

        try:
            self._pooling_function = {
                'max': lambda x: np.max(x, axis=0),
                'mean': lambda x: np.mean(x, axis=0),
                'min': lambda x: np.min(x, axis=0)}[pooling]
        except KeyError:
            raise ValueError('pooling method must be "max", "min" or "mean"')

    def get_files(self, dataset, word):
        """Returns the filenames in `gold` of given `dataset` and `word`

        Parameters
        ----------
        dataset: str
            The dataset type the `word` belongs to, must be 'synthetic' or
            'librispeech'.
        word : str
            The word to look after in the gold.

        """
        cond = (self._gold.type == dataset) & (self._gold.word == word)
        files = self._gold.filename[cond].to_list()
        assert 0 < len(files) <= 10
        return files

    # we use a cache because each key (aka word) will be called several time in
    # the distance calculation process, as a word can be part of several pairs.
    @functools.lru_cache(maxsize=None)
    def pooling(self, filename, dataset):
        """Loads `dataset/filename.txt` as an array and returns its pooling"""
        filename = self._folder / dataset / (filename + '.txt')
        return self._pooling_function(np.loadtxt(filename))

    def distance(self, pair):
        """Returns the mean distance between a pair of words"""
        # get the list of tokens corresponding to the given pair of words
        tokens_1 = self.get_files(pair.type, pair.word_1)
        tokens_2 = self.get_files(pair.type, pair.word_2)

        # compute the mean distance across all pairs of tokens after pooling
        return scipy.spatial.distance.cdist(
            np.asarray([self.pooling(x, pair.type) for x in tokens_1]),
            np.asarray([self.pooling(x, pair.type) for x in tokens_2]),
            metric=self._metric).mean()


def evaluate(gold_file, pairs_file, submission_dir, metric, pooling):
    """Returns the score on each words pair of the dataset

    Parameters
    ----------
    gold_file : path
        The gold file (csv format) for the semantic dataset (test or dev).
    pairs_file : path
        The pairs file (csv format) corresponding to `gold_file` (dev or test).
    submission_dir : path
        The submission directory containing the embeddings to evaluate.
    metric: str
        The metric to use for distance computation, must be a metreic supported
        by `scipy.spatial.distance.cdist` (see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    pooling : str
        The pooling method to use, must be 'min', 'max', or 'mean'.

    Returns
    -------
    pairs : pandas.DataFrame
        The same content as in `pairs_file` with an additional 'score' column
        containing the evaluated machine score for each pairs of words.

    Raises
    ------
    ValueError
        If one of the input parameters is not valid.
    OSError
        If a file defined in `gold_file` is not found in `submission_dir`.

    """
    # ensures input arguments are correct
    for input_file in (gold_file, pairs_file):
        if not pathlib.Path(input_file).is_file():
            raise ValueError(f'file not found: {input_file}')
    if not pathlib.Path(submission_dir).is_dir():
        raise ValueError(f'{submission_dir} is not a directory')

    # load the pairs file
    pairs = pandas.read_csv(pairs_file, header=0)

    # prepare for distance computations
    helper = EvaluationHelper(submission_dir, gold_file, metric, pooling)

    # compute distance for each pair in the dataset
    pairs['score'] = [helper.distance(pair) for _, pair in pairs.iterrows()]
    return pairs
