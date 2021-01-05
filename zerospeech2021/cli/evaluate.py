"""Evaluation program for ZR2021 submissions"""

import atexit
import os
import pathlib
import shutil
import sys
import tempfile
import zipfile

import click
import pandas
import yaml

from zerospeech2021 import phonetic, lexical, syntactic, semantic


def write_csv(frame, filename):
    frame.to_csv(filename, index=False, float_format='%.4f')
    print(f'  > Wrote {filename}')


def eval_lexical(dataset, submission, output, kinds):
    for kind in kinds:  # 'dev' or 'test'
        print(f'Evaluating lexical {kind}...')

        gold_file = dataset / 'lexical' / kind / 'gold.csv'
        submission_file = submission / 'lexical' / f'{kind}.txt'

        by_pair, by_frequency, by_length = lexical.evaluate(
            gold_file, submission_file)

        write_csv(
            by_pair, output / f'score_lexical_{kind}_by_pair.csv')
        write_csv(
            by_frequency, output / f'score_lexical_{kind}_by_frequency.csv')
        write_csv(
            by_length, output / f'score_lexical_{kind}_by_length.csv')


def eval_semantic(dataset, submission, output, kinds, njobs):
    # load metric and poling parameters from meta.yaml
    meta = yaml.safe_load((submission / 'meta.yaml').open('r').read())
    metric = meta['parameters']['semantic']['metric']
    pooling = meta['parameters']['semantic']['pooling']

    for kind in kinds:  # 'dev' or 'test'
        print(f'Evaluating semantic {kind} '
              f'(metric={metric}, pooling={pooling})...')

        gold_file = dataset / 'semantic' / kind / 'gold.csv'
        pairs_file = dataset / 'semantic' / kind / 'pairs.csv'
        pairs, correlation = semantic.evaluate(
            gold_file, pairs_file, submission / 'semantic' / kind,
            metric, pooling, njobs=njobs)

        write_csv(
            pairs, output / f'score_semantic_{kind}_pairs.csv')
        write_csv(
            correlation, output / f'score_semantic_{kind}_correlation.csv')


def eval_syntactic(dataset, submission, output, kinds):
    for kind in kinds:  # 'dev' or 'test'
        print(f'Evaluating syntactic {kind}...')

        gold_file = dataset / 'syntactic' / kind / 'gold.csv'
        submission_file = submission / 'syntactic' / f'{kind}.txt'

        by_pair, by_type = syntactic.evaluate(gold_file, submission_file)

        write_csv(
            by_pair, output / f'score_syntactic_{kind}_by_pair.csv')
        write_csv(
            by_type, output / f'score_syntactic_{kind}_by_type.csv')


def eval_phonetic(dataset, submission, output, kinds, force_cpu):
    meta = yaml.safe_load((submission / 'meta.yaml').open('r').read())
    metric = meta['parameters']['phonetic']['metric']
    frame_shift = meta['parameters']['phonetic']['frame_shift']

    results = []
    for kind in kinds:  # 'dev' or 'test'
        results.append(phonetic.evaluate(
            submission / 'phonetic', dataset / 'phonetic',
            kind, metric, frame_shift, force_cpu=force_cpu))

    write_csv(pandas.concat(results), output / 'score_phonetic.csv')


@click.command(epilog='See https://zerospeech.com/2021 for more details')
@click.argument('dataset', type=pathlib.Path)
@click.argument('submission', type=pathlib.Path)
@click.option(
    '-j', '--njobs', default=1, type=int,
    help='Parallel jobs to use for semantic part (default to 1)')
@click.option(
    '--force-cpu', help='Do not use GPU for phonetic part', is_flag=True)
@click.option(
    '-o', '--output-directory', type=pathlib.Path,
    default='.', show_default=True,
    help="Directory to store output results")
@click.option('--no-phonetic', help="Skip phonetic part", is_flag=True)
@click.option('--no-lexical', help="Skip lexical part", is_flag=True)
@click.option('--no-syntactic', help="Skip syntactic part", is_flag=True)
@click.option('--no-semantic', help="Skip semantic part", is_flag=True)
def evaluate(
        dataset, submission, njobs, force_cpu, output_directory,
        no_phonetic, no_lexical, no_syntactic, no_semantic):
    """Evaluate a submission to the Zero Resource Speech Challenge 2021

    DATASET is the root directory of the ZR2021 dataset, as downloaded from
    https://zerospeech.com/2021.

    SUBMISSION is the submission to evaluate, it can be a .zip file or a
    directory.

    """
    try:
        # regular participants can only evaluate dev datasets, test can only be
        # evaluated by doing an official submission to the challenge. The
        # ZEROSPEECH2021_TEST_GOLD environment variable is used by organizers
        # to provide test gold files to the evaluation program while keeping
        # the program as simple as possible to participants.
        kinds = ['dev']
        if 'ZEROSPEECH2021_TEST_GOLD' in os.environ:
            kinds.append('test')
            dataset = pathlib.Path(os.environ['ZEROSPEECH2021_TEST_GOLD'])

        # ensures the dataset exists
        dataset = dataset.resolve(strict=True)
        if not dataset.is_dir():
            raise ValueError(f'dataset not found: {dataset}')

        # ensures the submission exists, it it is a zip, uncompress it
        submission = submission.resolve(strict=True)
        if submission.is_file() and zipfile.is_zipfile(submission):
            # create a temp directory we remove at exit
            submission_unzip = tempfile.mkdtemp()
            atexit.register(shutil.rmtree, submission_unzip)

            # uncompress to the temp directory
            print(f'Unzip submission to {submission_unzip}...')
            zipfile.ZipFile(submission, 'r').extractall(submission_unzip)
            submission = pathlib.Path(submission_unzip)
        elif not submission.is_dir():
            raise ValueError(
                f'submssion is not a zip file or a directory: {submission}')

        if not output_directory.is_dir():
            output_directory.mkdir(exist_ok=True, parents=True)

        if not no_lexical:
            eval_lexical(dataset, submission, output_directory, kinds)

        if not no_semantic:
            eval_semantic(dataset, submission, output_directory, kinds, njobs)

        if not no_syntactic:
            eval_syntactic(dataset, submission, output_directory, kinds)

        if not no_phonetic:
            eval_phonetic(
                dataset, submission, output_directory, kinds, force_cpu)
    except ValueError as error:
        print(f'ERROR: {error}')
        sys.exit(-1)
