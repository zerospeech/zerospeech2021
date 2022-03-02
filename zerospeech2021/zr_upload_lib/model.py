import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

SERVER_LOCATION: str = "https://api.zerospeech.com"
CLIENT_ID: str = "cli_uploader"
CLIENT_SECRET: str = 'TaX9K1WtryizOTr5pLUM4OoqXZE5QGlj3Xo6dkh3CcI='
NB_RETRY_ATTEMPTS: int = 2
MULTIPART_THRESHOLD: int = 500000000  # in bytes (500MB)
AUTH_FILE: str = "~/.zerospeech-token"
CHALLENGE_ID = 7


def get_challenge_id():
    """ Get the current challenge id from the current environment or return the default. """
    return os.environ.get("CHALLENGE_ID", CHALLENGE_ID)


class ZrApiException(Exception):
    pass


@dataclass
class ManifestFileIndexItem:
    """ Upload File Manifest Item """
    file_name: str
    file_size: int
    file_hash: Optional[str] = None

    def dict(self):
        return {f"{x}": getattr(self, x) for x in self.__dataclass_fields__.keys()}

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class SplitManifest:
    """ A class containing information about archive split"""
    filename: str
    tmp_location: Path
    hash: str
    index: Optional[List[ManifestFileIndexItem]]
    multipart: bool = True
    hashed_parts: bool = True
    completed: int = 0

    def dict(self):
        data = {f"{x}": f"{getattr(self, x)}" for x in self.__dataclass_fields__.keys()}
        if "index" in data.keys():
            data["index"] = [
                item.dict() for item in self.index
            ]

        return data

    @classmethod
    def from_dict(cls, data):
        if "index" in data.keys():
            data["index"] = [
                ManifestFileIndexItem.from_dict(item) for item in data["index"]
            ]
        return cls(**data)


class UploadManifest:
    """ Fail-safe multi-part upload"""

    @classmethod
    def load(cls, filename: Path, retries: int = 2):
        with filename.open('r') as fp:
            dd = json.load(fp)
        return cls(dd["manifest"], filename, metadata=dd["metadata"], retries=retries)

    def __init__(self, list_manifest, save_file: Path, metadata=None, retries: int = 2):
        if isinstance(list_manifest, dict):
            self.man = list_manifest
        else:
            self.man = {
                f"{name}": 'todo'
                for name in list_manifest
            }
        self.save_file = save_file
        self.retries = retries
        if metadata:
            self._metadata = metadata
        else:
            self._metadata = {}
        self.save()

    def __iter__(self):
        return self

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, data):
        self._metadata.update(data)
        self.save()

    def __next__(self):
        for k, v in self.man.items():
            if v == 'todo':
                return k
        for k, v in self.man.items():
            if v == 'waiting':
                return k
        for k, v in self.man.items():
            if 'retry' in v:
                return k
        raise StopIteration

    def status(self, key):
        return self.man[key]

    def set_waiting(self, key):
        if self.man[key] == 'todo':
            self.man[key] = "waiting"
            self.save()

    def set_done(self, key):
        self.man[key] = "done"
        self.save()

    def set_failed(self, key):
        k = self.man[key]
        if k in ["waiting", "todo"]:
            self.man[key] = "retry_1"
        elif "retry" in k:
            nb = int(k.split('_')[1])
            nb += 1
            if nb > self.retries:
                st = 'failed'
            else:
                st = f"retry_{nb}"
            self.man[key] = st
        self.save()

    def save(self):
        with self.save_file.open('w') as fp:
            json.dump({
                "manifest": self.man,
                "metadata": self.metadata
            }, fp)

    def is_complete(self):
        for k, v in self.man.items():
            if v != "done":
                return False
        return True

    def get_failed(self):
        return [k for k, v in self.man.items() if v == 'failed']

    def clear(self):
        # remove checkpoint file
        self.save_file.unlink()
