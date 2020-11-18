"""Semantic part of the ZR2021 (validation and evaluation)"""

import pathlib

import numpy as np
import pandas
import scipy.spatial
import scipy.stats
import joblib

from zerospeech2021.exception import (
    MismatchError, FileFormatError, ValidationError, EntryMissingError)


def _validate_file(source_file, submission):
    """Verifies that a feature file is a 2D numpy array of floats

    :param source_file: input file
    :param submission: location of submitted files
    :return: a pair (error, ncols)

    """
    try:
        target_file = submission / (source_file + '.txt')
        if not target_file.is_file():
            raise EntryMissingError(source=source_file, expected=target_file)

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


def _get_files(gold, dataset, word):
    """Returns the filenames in `gold` of given `dataset` and `word`

    Parameters
    ----------
    dataset: str
        The dataset type the `word` belongs to, must be 'synthetic' or
        'librispeech'.
    word : str
        The word to look after in the gold.

    """
    cond = (gold['type'] == dataset) & (gold['word'] == word)
    files = gold['filename'][cond].to_list()
    assert 0 < len(files) <= 10
    return files


def _compute_distance(pair, gold, pool, metric):
    """Returns the mean distance between a pair of words"""
    # get the list of tokens corresponding to the given pair of words
    tokens_1 = _get_files(gold, pair['type'], pair['word_1'])
    tokens_2 = _get_files(gold, pair['type'], pair['word_2'])

    X = np.asarray(pool[pool['filename'].isin(tokens_1)]['pooling'].tolist())
    Y = np.asarray(pool[pool['filename'].isin(tokens_2)]['pooling'].tolist())

    # print(f'{pair.word_1}: {tokens_1}, {X.shape}')
    # print(f'{pair.word_2}: {tokens_2}, {Y.shape}')

    # compute the mean distance across all pairs of tokens after pooling
    return scipy.spatial.distance.cdist(X, Y, metric=metric).mean()


def _correlation(df):
    # choose 'similarity' or 'relatedness' column (the one with no NaN)
    human = df.similarity if df.relatedness.hasnans else df.relatedness
    assert not human.hasnans

    # return spearman correlation and pvalue
    return scipy.stats.spearmanr(human.to_numpy(), df.score.to_numpy())


def _compute_correlation(pairs):
    """"Returns the Spearman's correlation between human and machine scores"""
    # for each (type/dataset) combination, compute spearman correlation and pvalue
    serie = pairs.groupby([pairs['type'], pairs['dataset']]).apply(_correlation)

    # transfrom raw result in a usable dataframe
    frame = serie.to_frame().rename(columns={0: 'spearman'}).reset_index()
    frame[['correlation', 'pvalue']] = pandas.DataFrame(frame.spearman.tolist())
    return frame.drop(['spearman'], axis=1)


def evaluate(gold_file, pairs_file, submission_dir, metric, pooling, njobs=1):
    # ensures input arguments are correct
    for input_file in (gold_file, pairs_file):
        if not pathlib.Path(input_file).is_file():
            raise ValueError(f'file not found: {input_file}')
    if not pathlib.Path(submission_dir).is_dir():
        raise ValueError(f'{submission_dir} is not a directory')

    # get the pooling function
    try:
        _pooling_function = {
            'max': lambda x: np.max(x, axis=0),
            'mean': lambda x: np.mean(x, axis=0),
            'min': lambda x: np.min(x, axis=0)}[pooling]
    except KeyError:
        raise ValueError('pooling method must be "max", "min" or "mean"')

    # load the pairs and gold files
    pairs = pandas.read_csv(pairs_file, header=0)
    gold = pandas.read_csv(gold_file, header=0)

    # a data frame [filename, type, pooling] computed in parallel
    print(f'  > Computing {pooling} pooling...')
    pool = pandas.DataFrame(
        joblib.Parallel(n_jobs=njobs)(
            joblib.delayed(
                lambda x: (x[1], x[0], _pooling_function(
                    np.loadtxt(submission_dir / x[0] / (x[1] + '.txt')))))
            (x) for _, x in gold.iterrows()),
        columns=['filename', 'type', 'pooling'])

    print(f'  > Computing {metric} distances...')
    pairs['score'] = [
        _compute_distance(pair, gold, pool, metric)
        for _, pair in pairs.iterrows()]

    # compute correlations
    print('  > Computing Spearman correlations')
    correlation = _compute_correlation(pairs)
    return pairs, correlation
