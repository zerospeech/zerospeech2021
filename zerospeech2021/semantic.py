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


def _compute_distance(pair, gold, pool, metric):
    """Returns the mean distance between a pair of words"""
    function = {
        'librispeech': _compute_distance_librispeech,
        'synthetic': _compute_distance_synthetic}[pair['type']]

    return function(pair, gold, pool, metric)


def _compute_distance_librispeech(pair, gold, pool, metric):
    # filter out 'synthetic' data from gold
    assert pair['type'] == 'librispeech'
    gold = gold[gold['type'] == 'librispeech']

    # get the list of tokens corresponding to the given pair of words
    tokens_1 = gold['filename'][gold['word'] == pair['word_1']]
    tokens_2 = gold['filename'][gold['word'] == pair['word_2']]
    assert 0 < len(tokens_1) <= 10 and 0 < len(tokens_2) <= 10

    X = np.asarray(pool[pool['filename'].isin(tokens_1)]['pooling'].tolist())
    Y = np.asarray(pool[pool['filename'].isin(tokens_2)]['pooling'].tolist())

    # compute the mean distance across all pairs of tokens after pooling
    return scipy.spatial.distance.cdist(X, Y, metric=metric).mean()


def _compute_distance_synthetic(pair, gold, pool, metric):
    # filter out 'librispeech' data from gold
    assert pair['type'] == 'synthetic'
    gold = gold[gold['type'] == 'synthetic']

    # get the list of tokens corresponding to the given pair of words
    tokens_1 = gold[['filename', 'voice']][gold['word'] == pair['word_1']]
    tokens_2 = gold[['filename', 'voice']][gold['word'] == pair['word_2']]
    tokens = tokens_1.merge(tokens_2, on='voice').drop(['voice'], axis=1)

    # compute the mean of distances within a given voice
    dist = 0
    for _, (filename_x, filename_y) in tokens.iterrows():
        X = pool[pool['filename'] == filename_x]['pooling'].item()
        Y = pool[pool['filename'] == filename_y]['pooling'].item()
        dist += scipy.spatial.distance.cdist(
            np.atleast_2d(X), np.atleast_2d(Y), metric=metric)[0][0]
    return dist / len(tokens)


def _correlation(df):
    # choose 'similarity' or 'relatedness' column (the one with no NaN)
    human = df.similarity if df.relatedness.hasnans else df.relatedness
    assert not human.hasnans

    # return spearman correlation. Humans score are similarity (high when
    # close) so we take the opposite to have a quantity close to a distance
    # (low when close)
    return 100 * scipy.stats.spearmanr(
        - human.to_numpy(), df.score.to_numpy())[0]


def _compute_correlation(pairs):
    """"Returns the Spearman's correlation between human and machine scores"""
    # for each (type/dataset) combination, compute spearman correlation
    serie = pairs.groupby([pairs['type'], pairs['dataset']]).apply(_correlation)

    # transfrom raw result in a usable dataframe
    return serie.to_frame().rename(columns={0: 'correlation'}).reset_index()


def evaluate(gold_file, pairs_file, submission_dir, metric, pooling, njobs=1):
    """Returns the distance of each words pair and overall correlations

    Parameters
    ----------
    gold_file : path
        The gold file (csv format) for the dev or test semantic dataset.
    pairs_file : path
        The pairs file (csv format) corresponding to `gold_file` (dev or test).
    submission_dir : path
        The submission directry containing the embeddings to evaluate.
    metric : str
        The metric to use for distance computation, must be a metric supported
        by `scipy.spatial.distance.cdist` (see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    pooling : str
        The pooling method to use, must be 'min', 'max', 'mean', 'sum', 'last',
        'lastlast' or 'off'.

    Returns
    -------
    pairs : pandas.DataFrame
        The same content as in `pairs_file` with an additional 'score' column
        containing the evaluated machine scores for each pair of words.
    correlation : pandas.DataFrame
        The Spearman correlation between human judgements and machine scores on
        each dataset. The frame contains the columns 'type', 'dataset' and
        'correlation'.

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

    # get the pooling function
    try:
        _pooling_function = {
            'max': lambda x: np.max(x, axis=0),
            'mean': lambda x: np.mean(x, axis=0),
            'min': lambda x: np.min(x, axis=0),
            'sum': lambda x: np.sum(x, axis=0),
            'last': lambda x: x[-1],
            'lastlast': lambda x: x[-2],
            'off': lambda x: x}[pooling]
    except KeyError:
        raise ValueError(
            'pooling method must be "max", "min", "mean", "sum", '
            '"last" or "lastlast"')

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
    print('  > Computing Spearman correlations...')
    correlation = _compute_correlation(pairs)
    return pairs, correlation
