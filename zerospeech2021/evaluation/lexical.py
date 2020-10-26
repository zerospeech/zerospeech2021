"""Lexical level evaluation for the ZeroSpeech2021 challenge"""

import argparse
import pathlib
import pandas


def load_data(gold_file, submission_file):
    """Returns the data required for evaluation as a pandas data frame

    Each line of the returned data frame contains a pair (word, non word) and
    has the following columns: 'id', 'voice', 'frequency', 'word', 'word
    score', 'non word', 'score non word'.

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
        input_file = pathlib.Path(input_file)
        if not input_file.is_file():
            raise ValueError(f'file not found: {input_file}')

    # load them as data frames indexed by filenames
    gold = pandas.read_csv(gold_file, header=0, index_col='filename')
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
    data.drop(columns=['phones', 'filename'], inplace=True)

    # going from a word per line to a pair (word, non word) per line
    data = pandas.concat([
        data.loc[data['correct'] == 1].reset_index().rename(
            lambda x: 'w_' + x, axis=1),
        data.loc[data['correct'] == 0].reset_index().rename(
            lambda x: 'nw_' + x, axis=1)], axis=1)
    data.drop(
        ['w_index', 'nw_index', 'nw_voice', 'nw_frequency',
         'w_correct', 'nw_correct', 'nw_id'],
        axis=1, inplace=True)
    data.rename(
        {'w_id': 'id', 'w_voice': 'voice', 'w_frequency': 'frequency',
         'w_word': 'word', 'nw_word': 'non word',
         'w_score': 'score word', 'nw_score': 'score non word'},
        axis=1, inplace=True)

    return data


def evaluate_by_pair(data):
    """Returns a data frame with the computed scores by (word, non word) pair

    Parameters
    ----------
    data : pandas.DataFrame
        The result of `load_data`

    Returns
    -------
    by_pair : pandas.DataFrame
        The evaluated (word, non word) pairs, the data frame has the columns:
        'frequency', 'word', 'non word' and 'score'.

    Raise
    -----
    ValueError
        If the input files cannot be opened or in case of data mismatch between
        the two files.

    """
    # compute the score for each pair in an additional 'score' column, then
    # delete the 'score word' and 'score non word' columns that become useless
    score = data.loc[:, ['score word', 'score non word']].to_numpy()
    data['score'] = (
        0.5 * (score[:, 0] == score[:, 1])
        + (score[:, 0] > score[:, 1]))
    data.drop(columns=['score word', 'score non word'], inplace=True)

    # finally get the mean score across voices for al pairs
    score = data.groupby('id').apply(lambda x: (
        x.iat[0, 2],  # frequency
        x.iat[0, 3],  # word
        x.iat[0, 4],  # non word
        x['score'].mean()))
    return pandas.DataFrame(
        score.to_list(), columns=['frequency', 'word', 'non word', 'score'])


def evaluate_by_frequency(by_pair):
    """Returns a data frame with mean scores by frequency bands

    Parameters
    ----------
    by_pair: pandas.DataFrame
        The output of `evaluate_by_pair`

    Returns
    -------
    by_frequency : pandas.DataFrame
        The score collapsed on frequency bands, the data frame has the
        following columns: 'frequency', 'score'.

    """
    return by_pair.groupby('frequency').mean().reset_index()


def evaluate(gold_file, submission_file):
    """Returns the score by (word, non word) pair and by frequency band

    Parameters
    ----------
    gold_file : path
        The gold file (csv format) for the lexical dataset (test or dev).
    submission_file : path
        The submission corresponding to the provided gold file.

    Returns
    -------
    by_pair : pandas.DataFrame
        The evaluated (word, non word) pairs, the data frame has the columns:
        'frequency', 'word', 'non word' and 'score'.
    by_frequency : pandas.DataFrame
        The score collapsed on frequency bands, the data frame has the
        following columns: 'frequency', 'score'.

    Raise
    -----
    ValueError
        If the input files cannot be opened or in case of data mismatch between
        the two files.

    """
    data = load_data(gold_file, submission_file)
    by_pair = evaluate_by_pair(data)
    by_frequency = evaluate_by_frequency(by_pair)
    return by_pair, by_frequency


def main():
    """CLI for lexical evaluation"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'gold_file', type=pathlib.Path)
    parser.add_argument(
        'submission_file', type=pathlib.Path)
    parser.add_argument(
        '-o', '--output-directory', default='.', type=pathlib.Path)
    args = parser.parse_args()

    by_pair, by_frequency = evaluate(args.gold_file, args.submission_file)

    by_pair.to_csv(
        args.output_directory / 'score_lexical_by_pair.csv', index=False)
    by_frequency.to_csv(
        args.output_directory / 'score_lexical_by_frequency.csv', index=False)


if __name__ == '__main__':
    main()
