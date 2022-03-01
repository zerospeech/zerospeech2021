from pathlib import Path

import requests

from rich.console import Console
from rich.table import Table


from . import model

console = Console()


def view_challenges():
    """ Fetches the list of available challenges and allows selecting one."""
    response = requests.get(
        f"{model.SERVER_LOCATION}/challenges/", params={"include_inactive": "false"})
    if response.status_code != 200:
        raise ValueError('Request to server Failed !!')

    challenges = response.json()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Challenge")
    table.add_column("ID")

    for item in challenges:
        table.add_row(f"{item.get('label', '')}", f"{item.get('id', 'XX')}")

    console.print(table)


def create_multipart_submission(challenge_id: int, file_meta: dict, _token: str):
    """ Create a multipart upload submission session on the server via the API."""
    data = {
        "filename": file_meta["filename"],
        "hash": file_meta["hash"],
        "multipart": True,
        "index": file_meta['index']
    }

    return requests.post(
        f'{model.SERVER_LOCATION}/challenges/{challenge_id}/submission/create',
        json=data,
        headers={
            'Authorization': f'Bearer {_token}'
        })


def create_single_part_submission(challenge_id: int, filename: Path, _hash: str, _token: str):
    """ Create a single part submission upload session on the server via the API."""
    return requests.post(
        f'{model.SERVER_LOCATION}/challenges/{challenge_id}/submission/create',
        json={
            "filename": f"{filename}",
            "hash": _hash,
            "multipart": False,
        },
        headers={
            'Authorization': f'Bearer {_token}'
        })


def submission_upload(challenge_id: int, submission_id: str, file: Path, _token: str):
    """Upload a file (or part) to an existing upload session."""
    response = requests.put(
        f'{model.SERVER_LOCATION}/challenges/{challenge_id}/submission/upload',
        params={
            "part_name": file.name,
            "submission_id": f"{submission_id}"
        },
        files={f'file_data': file.open('rb').read()},
        headers={
            'Authorization': f'Bearer {_token}'
        }
    )
    return response
