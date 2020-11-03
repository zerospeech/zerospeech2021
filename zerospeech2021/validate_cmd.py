import click


@click.group()
def validate():
    """ Command to validate zerospeech2021 submissions """
    pass


@validate.command()
@click.argument('submission_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument('dataset_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
def all(submission_directory, dataset_directory):
    """ Validate all zerospeech 2021 taks """
    # todo call validation
    pass


@validate.command()
@click.argument('submission_file', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.argument('dataset_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
def lexical(submission_file, dataset_directory):
    """ Validate lexical submission """
    # todo call validation
    pass


@validate.command()
@click.argument('submission_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument('dataset_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
def semantic(submission_directory, dataset_directory):
    """ Validate semantic submission """
    # todo call validation
    pass


@validate.command()
@click.argument('submission_file', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.argument('dataset_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
def syntactic(submission_file, dataset_directory):
    """ Validate syntactic submission """
    # todo call validation
    pass


@validate.command()
@click.argument('submission_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument('dataset_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
def phonetic(submission_directory, dataset_directory):
    """ Validate phonetic submission """
    # todo call validation
    pass


def run():
    validate()


if __name__ == '__main__':
    run()
