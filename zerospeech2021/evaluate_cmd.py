import click


@click.group()
def evaluate():
    """ Command to validate zerospeech2021 submissions """
    pass


@evaluate.command()
@click.argument('gold_file', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.argument('submission_file', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-o', '--output-directory', 'output', type=click.Path(file_okay=False, dir_okay=True, exists=True),
              help="Location to output results")
def lexical(gold_file, submission_file, output):
    """ Validate lexical submission """
    # todo call validation
    pass


@evaluate.command()
@click.argument('gold_file', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.argument('submission_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.option('-o', '--output-directory', 'output', type=click.Path(file_okay=False, dir_okay=True, exists=True),
             help="Location to output results")
def semantic(gold_file, submission_directory, output):
    """ Validate semantic submission """
    # todo call validation
    pass


@evaluate.command()
@click.argument('gold_file', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.argument('submission_file', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-o', '--output-directory', 'output', type=click.Path(file_okay=False, dir_okay=True, exists=True),
              help="Location to output results")
def syntactic(gold_file, submission_file, output):
    """ Validate syntactic submission """
    # todo call validation
    pass


@evaluate.command()
@click.argument('gold_file', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.argument('submission_directory', type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.option('-o', '--output-directory', 'output', type=click.Path(file_okay=False, dir_okay=True, exists=True))
def phonetic(gold_file, submission_directory, output):
    """ Validate phonetic submission """
    # todo call validation
    pass


def run():
    evaluate()


if __name__ == '__main__':
    run()
