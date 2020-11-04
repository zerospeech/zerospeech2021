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
@click.argument('dataset', type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument('submission', type=click.Path(file_okay=True, dir_okay=True, exists=True))
@click.option('--lexical/--no-lexical', default=True, help="Validate lexical task",
              show_default=True)
@click.option('--semantic/--no-semantic', default=True, help="Validate semantic task",
              show_default=True)
@click.option('--syntactic/--no-syntactic', default=True, help="Validate syntactic task",
              show_default=True)
@click.option('--phonetic/--no-phonetic', default=True, help="Validate phonetic task",
              show_default=True)
def validate(**kwargs):
    """ Validate submission """
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
