import sys
from pathlib import Path

import requests
from rich.console import Console

from . import model

# Fancy console
console = Console()


def login(username: str, password: str):
    """ Create an auth session in zerospeech.com

    :returns: token<str> token used to authentify the current session
    """

    # request login from server
    response = requests.post(
        f'{model.SERVER_LOCATION}/auth/login',
        data={
            "grant_type": "password",
            "username": username,
            "password": password,
            "scopes": [],
            "client_id": model.CLIENT_ID,
            "client_secret": model.CLIENT_SECRET
        }
    )
    if response.status_code != 200:
        console.print(f"[red]:x:{response.status_code}[/red]: {response.json().get('detail')}")
        sys.exit(1)

    return response.json().get("access_token")


def logout(_token):
    """ Clears the given auth session on the back-end """
    return requests.delete(
        f'{model.SERVER_LOCATION}/auth/logout',
        headers={
            'Authorization': f'Bearer {_token}'
        })


def clear_session():
    """ Clear the current session locally and on the server."""
    token_file = Path(model.AUTH_FILE).expanduser().resolve()
    if token_file.is_file():
        with token_file.open() as fp:
            token = fp.read().replace("\n", "")

        # clear
        token_file.unlink(missing_ok=True)
        logout(token)
        console.print(f"Session saved @ {token_file} was removed.", style='green bold')


def create_session(token: str):
    """ Creates an new auth session & saves it locally """
    token_file = Path(model.AUTH_FILE).expanduser().resolve()

    with token_file.open('w') as fp:
        fp.write(token)


def get_session():
    """ Get or Create a new auth session """
    token_file = Path(model.AUTH_FILE).expanduser().resolve()

    if not token_file.is_file():
        console.print(f"No session found use login command to create one.", style='red  bold')
        sys.exit(1)

    with token_file.open() as fp:
        return fp.read().replace("\n", "")
