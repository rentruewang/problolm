# Copyright (c) ProBloLM Authors - All Rights Reserved

import typing

import github3
from github3 import GitHub
from github3.repos import Repository

from . import envs


@typing.no_type_check
def login() -> GitHub:
    token = envs.github_token()

    if session := github3.login(token=token):
        return session

    raise ValueError("Authentication failed with your `github_token`.")


@typing.no_type_check
def repo() -> Repository:
    owner, repo = envs.github_repo().split("/")

    if repo := login().repository(owner=owner, repository=repo):
        return repo

    raise ValueError("Reponsitory is not found. Most likely an authentication issue.")
