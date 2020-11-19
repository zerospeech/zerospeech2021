"""Validation program for ZR2021 submissions"""

import atexit
import pathlib
import shutil
import sys
import tempfile
import zipfile

import click

from zerospeech2021 import (
    exception, meta, phonetic, lexical, syntactic, semantic)


def _validate_directory(directory, expected):
    """Ensures the expected content is present in the directory"""
    expected = set(expected)
    observed = set(
        str(f.relative_to(directory))
        for f in pathlib.Path(directory).glob('*'))

    if expected != observed:
        raise exception.MismatchError(
            f'mismatch in directory {directory}', expected, observed)


def _validate_phonetic(submission, dataset, only_dev, njobs):
    print('Validating phonetic...')
    _validate_directory(
        submission / 'phonetic',
        ['dev-clean', 'dev-other'] if only_dev
        else ['dev-clean', 'dev-other', 'test-clean', 'test-other'])

    print('  > phonetic/dev')
    phonetic.validate(
        submission / 'phonetic',
        dataset / 'phonetic', 'dev',
        njobs=njobs)

    if not only_dev:
        print('  > phonetic/test')
        phonetic.validate(
            submission / 'phonetic',
            dataset / 'phonetic', 'test',
            njobs=njobs)


def _validate_lexical(submission, dataset, only_dev):
    print('Validating lexical...')
    _validate_directory(
        submission / 'lexical',
        ['dev.txt'] if only_dev else ['dev.txt', 'test.txt'])

    print('  > lexical/dev')
    lexical.validate(
        submission / 'lexical' / 'dev.txt',
        dataset, 'dev')

    if not only_dev:
        print('  > lexical/test')
        lexical.validate(
            submission / 'lexical' / 'test.txt',
            dataset, 'test')


def _validate_syntactic(submission, dataset, only_dev):
    print('Validating syntactic...')
    _validate_directory(
        submission / 'syntactic',
        ['dev.txt'] if only_dev else ['dev.txt', 'test.txt'])

    print('  > syntactic/dev')
    syntactic.validate(
        submission / 'syntactic' / 'dev.txt',
        dataset, 'dev')

    if not only_dev:
        print('  > syntactic/test')
        syntactic.validate(
            submission / 'syntactic' / 'test.txt',
            dataset, 'test')


def _validate_semantic(submission, dataset, only_dev, njobs):
    print('Validating semantic...')
    semantic_content = ['dev'] if only_dev else ['dev', 'test']
    _validate_directory(submission / 'semantic', semantic_content)

    for subdir in semantic_content:
        _validate_directory(
            submission / 'semantic' / subdir,
            ['librispeech', 'synthetic'])

    print('  > semantic/dev/synthetic')
    semantic.validate(
        submission / 'semantic', dataset, 'dev', 'synthetic', njobs=njobs)

    print('  > semantic/dev/librispeech')
    semantic.validate(
        submission / 'semantic', dataset, 'dev', 'librispeech', njobs=njobs)

    if not only_dev:
        print('  > semantic/test/synthetic')
        semantic.validate(
            submission / 'semantic', dataset, 'test', 'synthetic', njobs=njobs)

        print('  > semantic/test/librispeech')
        semantic.validate(
            submission / 'semantic', dataset, 'test', 'librispeech', njobs=njobs)


@click.command(epilog='See https://zerospeech.com/2021 for more details')
@click.argument('dataset', type=pathlib.Path)
@click.argument('submission', type=pathlib.Path)
@click.option(
    '-j', '--njobs', default=1, type=int,
    help='Number of parallel jobs (default to 1)')
@click.option('--only-dev', help='Skip test part', is_flag=True)
@click.option('--no-phonetic', help="Skip phonetic part", is_flag=True)
@click.option('--no-lexical', help="Skip lexical part", is_flag=True)
@click.option('--no-syntactic', help="Skip syntactic part", is_flag=True)
@click.option('--no-semantic', help="Skip semantic part", is_flag=True)
def validate(
        dataset, submission, njobs, only_dev,
        no_phonetic, no_lexical, no_syntactic, no_semantic):
    """Validate a submission to the Zero Resource Speech Challenge 2021

    DATASET is the root directory of the ZR2021 dataset, as downloaded with the
    zerospeech2021-download tool.

    SUBMISSION is the submission to validate, it can be a .zip file or a
    directory.

    """
    try:
        # ensures the dataset exists
        dataset = dataset.resolve(strict=True)
        if not dataset.is_dir():
            raise ValueError(f'dataset not found: {dataset}')

        # ensures the submission exists, it it is a zip, uncompress it
        submission = submission.resolve(strict=True)

        print('Prepare input...')
        print(f'  > dataset: {dataset}')
        print(f'  > submission: {submission}')

        if submission.is_file() and zipfile.is_zipfile(submission):
            # create a temp directory we remove at exit
            submission_unzip = tempfile.mkdtemp()
            atexit.register(shutil.rmtree, submission_unzip)

            # uncompress to the temp directory
            print(f'  > unzip submission to {submission_unzip}...')
            zipfile.ZipFile(submission, 'r').extractall(submission_unzip)
            submission = pathlib.Path(submission_unzip)
        elif not submission.is_dir():
            raise ValueError(
                f'submssion is not a zip file or a directory: {submission}')

        print('Validating root folder...')
        print('  > meta.yaml')
        is_open_source = meta.validate(submission)

        print('  > root folder')
        root_content = [
            'meta.yaml', 'phonetic', 'lexical', 'syntactic', 'semantic']
        if is_open_source:
            root_content.append('code')
        _validate_directory(submission, root_content)

        if is_open_source:
            if not (submission / 'code').is_dir():
                raise exception.ValidationError(
                    'submission specified as open source but '
                    'code folder is missing')
            if not list((submission / 'code').iterdir()):
                raise exception.ValidationError(
                    'submission specified as open source but '
                    'code folder is empty')
            print('  > code folder detected: submission will be manually '
                  'inspected to ensure it is open source')

        if not no_phonetic:
            _validate_phonetic(submission, dataset, only_dev, njobs)

        if not no_lexical:
            _validate_lexical(submission, dataset, only_dev)

        if not no_syntactic:
            _validate_syntactic(submission, dataset, only_dev)

        if not no_semantic:
            _validate_semantic(submission, dataset, only_dev, njobs)

    except (exception.ValidationError, ValueError, FileNotFoundError) as error:
        print(f'ERROR: {error}')
        print('Validation failed, please fix it and try again!')
        sys.exit(-1)

    print('Success!')
    sys.exit(0)
