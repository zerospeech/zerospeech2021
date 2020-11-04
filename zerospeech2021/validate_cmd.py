"""Validation program for ZR2021 submissions"""
import sys
from pathlib import Path
import click

from zerospeech2021 import exception, meta_file


def validate_lexical(dataset_location, submission_location):
    raise NotImplementedError()


def validate_syntactic(dataset_location, submission_location):
    from zerospeech2021.syntactic import validate

    submission_file = submission_location / 'syntactic_dev.txt'
    validate(submission_file, dataset_location, 'dev')

    submission_file = submission_location / 'syntactic_test.txt'
    validate(submission_file, dataset_location, 'test')


def validate_semantic(dataset_location, submission_location):
    raise NotImplementedError()


def validate_phonetic(dataset_location, submission_location):
    from zerospeech2021.phonetic import validation

    validation(submission_location, dataset_location, 'test')
    validation(submission_location, dataset_location, 'dev')


@click.command()
@click.argument(
    'dataset', type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument(
    'submission', type=click.Path(file_okay=True, dir_okay=True, exists=True))
@click.option(
    '--lexical/--no-lexical', help="Validate lexical task",
    default=True, show_default=True)
@click.option(
    '--semantic/--no-semantic', help="Validate semantic task",
    default=True, show_default=True)
@click.option(
    '--syntactic/--no-syntactic', help="Validate syntactic task",
    default=True, show_default=True)
@click.option(
    '--phonetic/--no-phonetic', help="Validate phonetic task",
    default=True, show_default=True)
def validate(**kwargs):
    """Validate a submission to the Zero Resource Speech Challenge 2021

    DATASET is the root directory of the ZR2021 dataset, as downloaded with
    the zerospeech2021-download tool.

    SUBMISSION is the submission to validate, it can be a .zip file or a
    directory.

    """
    dataset_location = Path(kwargs.get('dataset'))
    submission_location = Path(kwargs.get('submission'))
    try:
        if kwargs.get("lexical"):
            validate_lexical(dataset_location, submission_location)

        if kwargs.get("semantic"):
            validate_semantic(dataset_location, submission_location)

        if kwargs.get("syntactic"):
            validate_syntactic(dataset_location, submission_location)

        if kwargs.get("phonetic"):
            validate_phonetic(dataset_location, submission_location)

        meta_file.validate_meta_file(submission_location)
    except (exception.ValidationError, ValueError) as error:
        print(f'ERROR {error}')
        print('Validation failed, please fix it and try again!')
        sys.exit(-1)

    print('Validation success')
    sys.exit(0)
