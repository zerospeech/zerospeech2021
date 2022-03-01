import sys
from getpass import getpass
from pathlib import Path

import click

from rich.console import Console
from rich.progress import Progress, BarColumn

from zerospeech2021 import zr_upload_lib as zr_up

# Fancy console
console = Console()

# The challenge to use for uploads
CHALLENGE_ID: int = 1


@click.group(epilog='See https://zerospeech.com/2021 for more details')
@click.option('--debug', help="Print debug info", is_flag=True)
@click.pass_context
def upload_cmd(ctx, debug):
    ctx.debug = debug


@upload_cmd.command()
@click.option('-u', '--username', type=str)
@click.option('-p', '--password', type=str)
@click.option('--clear', is_flag=True)
@click.pass_obj
def login(debug, username, password, clear):
    # clear session
    if clear:
        zr_up.auth.clear_session()
        sys.exit(1)

    if not username:
        username = input('Username: ')

    if not password:
        password = getpass("Password: ")

    # login
    token = zr_up.auth.login(username, password)
    # save session
    zr_up.auth.create_session(token)
    console.print(f'Successfully logged in as {username}', style='green bold')


@upload_cmd.command()
@click.argument('archive_file', type=Path)
@click.pass_obj
def multipart(debug, archive_file):
    """ Upload an archive using multipart upload """
    if archive_file.is_file() and archive_file.suffix != ".zip":
        console.print(f"ERROR: given file: {archive_file} was not found or is not a .zip file !!",
                      style="red bold")
        sys.exit(1)

    # check if file is large enough for splitting
    will_split = archive_file.stat().st_size > zr_up.model.MULTIPART_THRESHOLD * 2

    checkpoint_file = archive_file.parent / f"{archive_file.stem}.checkpoint.json"
    zr_up.upload.ask_resume(checkpoint_file)
    token = zr_up.auth.get_session()

    with Progress(
            "[progress.description]{task.description}", BarColumn(),
    ) as progress:
        task = progress.add_task("[red]Uploading...", start=False, total=100)

        if will_split:
            zr_up.upload.multipart_upload(CHALLENGE_ID, archive_file, token, checkpoint_file)
        else:
            zr_up.upload.single_part_upload(CHALLENGE_ID, archive_file, token)

        progress.advance(task, advance=100)

    console.print(f"Successfully uploaded archive {archive_file} to zerospeech.com", style="green")


@upload_cmd.command()
@click.argument('archive_file', type=Path)
@click.pass_obj
def simple(debug, archive_file):
    """ Upload an archive using simple upload """
    if archive_file.is_file() and archive_file.suffix != ".zip":
        console.print(f"ERROR: given file: {archive_file} was not found or is not a .zip file !!",
                      style="red bold")
        sys.exit(1)

    token = zr_up.auth.get_session()
    with Progress(
            "[progress.description]{task.description}", BarColumn(),
    ) as progress:
        task = progress.add_task("[red]Uploading...", start=False, total=100)

        # upload
        zr_up.upload.single_part_upload(CHALLENGE_ID, archive_file, token)

        progress.advance(task, advance=100)

    console.print(f"Successfully uploaded archive {archive_file} to zerospeech.com", style="green")