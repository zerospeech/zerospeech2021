"""Syntactic part of the ZR2021 (validation and evaluation)"""

import collections
import pathlib

import pandas
from zerospeech2021.exception import FormatError, MismatchError


def _validate_line(index, line):
    """Auxiliary function to validate()

    Returns the filename in `line`, checks the score and raises FormatError if
    the line is not valid.

    """
    # ensure the line has two fields separated by a space
    line = line.strip()
    fields = line.split(' ')
    if len(fields) != 2:
        raise FormatError(
            index, f'must be "<filename> <score>" but is "{line}"')

    filename, score = tuple(fields)

    # ensure the second field is a positive float
    try:
        float(score)
    except ValueError:
        raise FormatError(
            index, f'<score> must be a float but is "{score}"')

    return filename


def validate(submission, dataset, kind):
    """Raises a ValidationError if the `submisison` file is not valid

    * The submission file must be in text format, each line as:
          <filename> <score>

    * The <filename> is the name of a wav file in the syntactic dataset,
      without path nor extension ("xKtnLJYiWGt", not
      "syntactic/dev/xKtnLJYiWGt.wav")

    * The <score> is a positive float

    Parameters
    ----------
    submisison: path
        The submisison file to validate, each line must be formatted as
        "<filename> <score>".
    dataset: path
        The root path of the ZR2021 dataset
    kind: str, optional
        Must be 'dev' or 'test'

    Raises
    ------
    ValueError
        If `kind` is not 'dev' or 'test', if `submisison` is not a file or if
        the dataset is not an existing directory.
    ValidationError
        If one line of the submisison file is not valid or if the submitted
        filenames does not fit the required ones.

    """
    if kind not in ('dev', 'test'):
        raise ValueError(
            f'kind must be "dev" or "test", it is {kind}')

    if not pathlib.Path(submission).is_file():
        raise ValueError(
            f'{kind} submission file not found: {submission}')

    # retrieve the required filenames that must be present in the submission
    dataset = pathlib.Path(dataset) / f'syntactic/{kind}'
    if not dataset.is_dir():
        raise ValueError(f'dataset not found: {dataset}')
    required_files = set(w.stem for w in dataset.glob('*.wav'))

    # ensure each line in the submission is valid and retrieve the filenames
    submitted_files = list(
        _validate_line(index + 1, line)
        for index, line in enumerate(open(submission, 'r')))

    # ensures the is no duplicate in the filenames
    duplicates = [
        f for f, n in collections.Counter(submitted_files).items() if n > 1]
    if duplicates:
        raise MismatchError('duplicates found', [], duplicates)

    # ensure all the required files are here and there is no extra filename
    if required_files != set(submitted_files):
        raise MismatchError(
            'mismatch in filenames', required_files, submitted_files)


def load_data(gold_file, submission_file):
    """Returns the data required for evaluation as a pandas data frame

    Each line of the returned data frame contains a pair of (correct,
    incorrect) sentences and has the following columns: 'id', 'voice', 'type',
    'sentence', 'score sentence', 'non sentence', 'score non sentence'.

    Parameters
    ----------
    gold_file : path
        The gold file for the lexical dataset (test or dev).
    submission_file : path
        The submission corresponding to the provided gold file.

    Returns
    -------
    data : pandas.DataFrame
        The data ready for evaluation

    Raise
    -----
    ValueError
        If the input files cannot be opened or in case of data mismatch between
        the two files.

    """
    # ensures the two input files are here
    for input_file in (gold_file, submission_file):
        if not pathlib.Path(input_file).is_file():
            raise ValueError(f'file not found: {input_file}')

    # load them as data frames indexed by filenames
    gold = pandas.read_csv(
        gold_file, header=0, index_col='filename')
    score = pandas.read_csv(
        submission_file, sep=' ', header=None,
        names=['filename', 'score'], index_col='filename')

    # ensures the filenames in gold and submission are the same
    if set(gold.index) != set(score.index):
        raise ValueError('mismatch between gold and submission !')

    # merge the gold and score using filenames, then remove the columns
    # 'phones' and 'filename' as we don't use them for evaluation
    data = pandas.concat([gold, score], axis=1)
    data.reset_index(inplace=True)
    data.drop(columns=['index'], inplace=True)

    # going from a word per line to a pair (word, non word) per line
    data = pandas.concat([
        data.loc[data['correct'] == 1].reset_index().rename(
            lambda x: 's_' + x, axis=1),
        data.loc[data['correct'] == 0].reset_index().rename(
            lambda x: 'ns_' + x, axis=1)], axis=1)
    data.drop(
        ['s_index', 'ns_index', 'ns_voice', 'ns_type', 'ns_subtype',
         's_correct', 'ns_correct', 'ns_id'],
        axis=1, inplace=True)

    data.rename(
        {'s_id': 'id',
         's_voice': 'voice',
         's_type': 'type',
         's_subtype': 'subtype',
         's_transcription': 'sentence',
         'ns_transcription': 'non sentence',
         's_score': 'score sentence',
         'ns_score': 'score non sentence'},
        axis=1, inplace=True)

    return data


def evaluate_by_pair(data):
    """Returns a data frame with the scores by (sentence, non sentence) pair

    Parameters
    ----------
    data : pandas.DataFrame
        The result of `load_data`

    Returns
    -------
    by_pair : pandas.DataFrame
        The evaluated (sentence, non sentence) pairs, the data frame has the
        columns: 'sentence', 'non sentence' 'type' and 'score'.

    """
    # compute the score for each pair in an additional 'score' column, then
    # delete the 'score word' and 'score non word' columns that become useless
    score = data.loc[:, ['score sentence', 'score non sentence']].to_numpy()
    data['score'] = (
        0.5 * (score[:, 0] == score[:, 1])
        + (score[:, 0] > score[:, 1]))
    data.drop(columns=['score sentence', 'score non sentence'], inplace=True)

    # finally get the mean score across voices for all pairs
    score = data.groupby(['type', 'subtype', 'id']).apply(lambda x: (
        x.iat[0, 2],  # type
        x.iat[0, 3],  # subtype
        x.iat[0, 4],  # sentence
        x.iat[0, 5],  # non sentence
        x['score'].mean()))
    return pandas.DataFrame(
        score.to_list(),
        columns=['type', 'subtype', 'sentence', 'non sentence', 'score'])


def evaluate_by_type(by_pair):
    """Returns a data frame with mean scores by syntax error type

    Parameters
    ----------
    by_pair: pandas.DataFrame
        The output of `evaluate_by_pair`

    Returns
    -------
    by_type : pandas.DataFrame
        The score collapsed on word length, the data frame has the
        following columns: 'type', 'score'.

    """
    return by_pair.score.groupby([by_pair.type, by_pair.subtype]).agg(
        n='count', score='mean', std='std').reset_index()


def evaluate(gold_file, submission_file):
    """Returns the score by sentences pair and by syntax type

    Parameters
    ----------
    gold_file : path
        The gold file (csv format) for the lexical dataset (test or dev).
    submission_file : path
        The submission corresponding to the provided gold file.

    Returns
    -------
    by_pair : pandas.DataFrame
        The evaluated pairs, the data frame has the columns:
        'sentence', 'non sentence' and 'score'.
    by_type : pandas.DataFrame
        The score collapsed on syntax errors types, the data frame has the
        following columns: 'type', 'score'.

    Raise
    -----
    ValueError
        If the input files cannot be opened or in case of data mismatch between
        the two files.

    """
    data = load_data(gold_file, submission_file)
    print(data.head())

    by_pair = evaluate_by_pair(data)
    print(by_pair.head())
    by_type = evaluate_by_type(by_pair)
    by_pair.drop(['type', 'subtype'], axis=1, inplace=True)

    return by_pair, by_type
