import shutil
import sys
from pathlib import Path

from rich import inspect, print
from rich.console import Console
from rich.progress import Progress
from rich.prompt import Prompt

from . import model
from .api_fn import (
    create_multipart_submission, submission_upload, create_single_part_submission
)
from .split import split_zip_v2, md5sum

# Fancy console
console = Console()


def multipart_upload(challenge_id: int, zipfile: Path, _token: str, checkpoint: Path):
    print("preparing metadata....")

    # check for checkpoint
    if checkpoint.is_file():
        file_list = model.UploadManifest.load(checkpoint, retries=model.NB_RETRY_ATTEMPTS)
        tmp_location = Path(file_list.metadata.get("tmp_location"))
        _token = file_list.metadata.get('token')
        challenge_id = file_list.metadata.get("challenge_id")
    else:
        manifest = split_zip_v2(zipfile)
        file_list = [i.file_name for i in manifest.index]
        tmp_location = manifest.tmp_location
        meta = {
            "tmp_location": f"{tmp_location}",
            "filename": manifest.filename,
            "hash": manifest.hash,
            "index": [i.dict() for i in manifest.index],
            "token": _token,
            "challenge_id": challenge_id
        }
        file_list = model.UploadManifest(file_list, checkpoint, meta, retries=model.NB_RETRY_ATTEMPTS)

    # check if submission session exists
    if "submission_id" in file_list.metadata:
        submission_id = file_list.metadata.get('submission_id')
    else:
        response = create_multipart_submission(challenge_id, file_list.metadata, _token)
        if response.status_code != 200:
            print(f'[red]:x:[/red][bold]Submission Creation Failed with code [red] {response.status_code}[/red][/bold]')
            inspect(response.json())
            sys.exit(1)

        submission_id = response.text.replace('"', '').replace("'", "")
        file_list.metadata = {"submission_id": submission_id}

    with Progress() as progress:
        task1 = progress.add_task("[red]Uploading parts...", total=len(file_list.man))

        for item in file_list:
            file_list.set_waiting(item)
            progress.update(task1, advance=0.5)
            file_path = tmp_location / item
            print(f'uploading : {file_path.name}...')
            response = submission_upload(
                challenge_id=challenge_id,
                submission_id=submission_id,
                file=file_path,
                _token=_token
            )

            if response.status_code == 200:
                print(f'[green]:heavy_check_mark: {file_path}')
                file_list.set_done(item)
                progress.update(task1, advance=0.5)
            else:
                progress.update(task1, advance=-0.5)
                file_list.set_failed(item)

    if file_list.is_complete():
        checkpoint.unlink()
        shutil.rmtree(tmp_location)
        return []
    else:
        return file_list.get_failed()


def single_part_upload(challenge_id: int, zipfile: Path, _token: str):
    zip_hash = md5sum(zipfile)
    response = create_single_part_submission(challenge_id, filename=zipfile, _hash=zip_hash, _token=_token)

    if response.status_code != 200:
        print(f'[red]:x:[/red][bold]Submission Creation Failed with code [red] {response.status_code}[/red][/bold]')
        inspect(response.json())
        sys.exit(1)

    submission_id = response.text.replace('"', '').replace("'", "")
    print(f'submission id: {submission_id}')
    response = submission_upload(
        challenge_id=challenge_id,
        submission_id=submission_id,
        file=zipfile,
        _token=_token
    )

    if response.status_code != 200:
        print(f'[red]:x:[/red][bold]Archive upload failed with code [red] {response.status_code}[/red][/bold]')
        print(response.json())
        sys.exit(1)


def ask_resume(file: Path):
    """ Ask the user to resume or not the upload """
    choice = "No"
    if file.is_file():
        choice = Prompt.ask("A checkpoint file was found. Do you wish to resume ?",
                            choices=["Yes", "No"])
        if choice == "No":
            file.unlink()

    return choice == "Yes"
