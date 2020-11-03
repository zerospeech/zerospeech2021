import sys
from pathlib import Path
import click

from zerospeech2021 import (
    exception,
    lexical as _lexical,
    syntactic as _syntactic,
    phonetic as _phonetic
)


@click.group()
def validate():
    """ Command to validate zerospeech2021 submissions """
    pass


@validate.command()
@click.argument('submission_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument('dataset_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
def all(submission_directory, dataset_directory):
    """ Validate all zerospeech 2021 tasks """
    try:
        pass
    except (exception.ValidationError, ValueError) as error:
        print(f'ERROR {error}')
        print('Validation failed, please fix it and try again!')
        sys.exit(-1)

    print('Validation success')
    sys.exit(0)


@validate.command()
@click.argument('submission_file', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.argument('dataset_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
def lexical(submission_file, dataset_directory):
    """ Validate lexical submission """
    try:
        pass
    except (exception.ValidationError, ValueError) as error:
        print(f'ERROR {error}')
        print('Validation failed, please fix it and try again!')
        sys.exit(-1)

    print('Validation success')
    sys.exit(0)


@validate.command()
@click.argument('submission_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument('dataset_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
def semantic(submission_directory, dataset_directory):
    """ Validate semantic submission """
    try:
        pass
    except (exception.ValidationError, ValueError) as error:
        print(f'ERROR {error}')
        print('Validation failed, please fix it and try again!')
        sys.exit(-1)

    print('Validation success')
    sys.exit(0)


@validate.command()
@click.argument('submission_file', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.argument('dataset_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
def syntactic(submission_file, dataset_directory):
    """ Validate syntactic submission """
    try:
        pass
    except (exception.ValidationError, ValueError) as error:
        print(f'ERROR {error}')
        print('Validation failed, please fix it and try again!')
        sys.exit(-1)

    print('Validation success')
    sys.exit(0)


@validate.command()
@click.argument('submission_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument('dataset_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.option('--dev/--no-dev', default=True, help="Validate dev set (dev-clean, dev-other)", show_default=True)
@click.option('--test/--no-test', default=True, help="Validate test set (test-clean, test-other)", show_default=True)
@click.option('--file-type', 'file_type', help='File type in the dataset', default='flac',
              type=click.Choice(['wav', 'flac'], case_sensitive=False), show_default=True)
def phonetic(submission_directory, dataset_directory, dev, test, file_type):
    """ Validate phonetic submission """
    submission_directory = Path(submission_directory)
    dataset_directory = Path(dataset_directory)

    try:
        if test:
            _phonetic.validation(
                submission_directory, dataset_directory, file_type, 'test'
            )

        if dev:
            _phonetic.validation(
                submission_directory, dataset_directory, file_type, 'dev'
            )
    except (exception.ValidationError, ValueError) as error:
        print(f'ERROR {error}')
        print('Validation failed, please fix it and try again!')
        sys.exit(-1)

    print('Validation success')
    sys.exit(0)


def run():
    validate()


if __name__ == '__main__':
    run()
