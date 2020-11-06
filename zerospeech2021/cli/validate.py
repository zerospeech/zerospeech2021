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


@click.command(epilog='See https://zerospeech.com/2021 for more details')
@click.argument('dataset', type=pathlib.Path)
@click.argument('submission', type=pathlib.Path)
@click.option('--only-dev', help='Skip test part', is_flag=True)
@click.option('--no-phonetic', help="Skip phonetic part", is_flag=True)
@click.option('--no-lexical', help="Skip lexical part", is_flag=True)
@click.option('--no-syntactic', help="Skip syntactic part", is_flag=True)
@click.option('--no-semantic', help="Skip semantic part", is_flag=True)
def validate(
        dataset, submission, only_dev,
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
        if submission.is_file() and zipfile.is_zipfile(submission):
            # create a temp directory we remove at exit
            submission_unzip = tempfile.mkdtemp()
            atexit.register(shutil.rmtree, submission_unzip)

            # uncompress to the temp directory
            print(f'Unzip submission ot {submission_unzip}...')
            zipfile.ZipFile(submission, 'r').extractall(submission_unzip)
            submission = submission_unzip
        elif not submission.is_dir():
            raise ValueError(
                f'submssion is not a zip file or a directory: {submission}')

        # validate meta.yaml
        meta.validate(submission)

        # validate phonetic
        if not no_phonetic:
            print('Validating phonetic dev...')
            phonetic.validate(
                submission / 'phonetic',
                dataset / 'phonetic', 'dev')

            if not only_dev:
                print('Validating phonetic test...')
                phonetic.validate(
                    submission / 'phonetic',
                    dataset / 'phonetic', 'test')

        # validate lexical
        if not no_lexical:
            print('Validating lexical dev...')
            lexical.validate(
                submission / 'lexical' / 'dev.txt',
                dataset, 'dev')

            if not only_dev:
                print('Validating lexical test...')
                lexical.validate(
                    submission / 'lexical' / 'test.txt',
                    dataset, 'test')

        # validate syntactic
        if not no_syntactic:
            print('Validating syntactic dev...')
            syntactic.validate(
                submission / 'syntactic' / 'dev.txt',
                dataset, 'dev')

            if not only_dev:
                print('Validating syntactic test...')
                syntactic.validate(
                    submission / 'syntactic' / 'test.txt',
                    dataset, 'test')

        # validate semantic
        if not no_semantic:
            print('Validating semantic dev...')
            semantic.validate(submission / 'semantic', dataset, 'dev')

            if not only_dev:
                print('Validating semantic test...')
                semantic.validate(submission / 'semantic', dataset, 'test')

    except (exception.ValidationError, ValueError, FileNotFoundError) as error:
        print(f'ERROR: {error}')
        print('Validation failed, please fix it and try again!')
        sys.exit(-1)

    print('Validation success')
    sys.exit(0)
