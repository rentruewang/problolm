# Copyright (c) ProBloLM Authors - All Rights Reserved

import json
import os

import requests


def get_pr_contents(repo: str, pr_number: int):
    # Build API URL
    commits_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/commits"

    # Request commit list
    resp = requests.get(
        commits_url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    resp.raise_for_status()
    commits = resp.json()

    print(f"Commits in PR #{pr_number}:")
    for c in commits:
        sha = c["sha"]
        msg = c["commit"]["message"]
        print(f"🧾 {sha}: {msg}")


if __name__ == "__main__":
    # Load event and PR number
    event = json.load(open(os.environ["GITHUB_EVENT_PATH"]))
    pr_number = event["pull_request"]["number"]

    repo = os.environ["REPO"]
    token = os.environ["GITHUB_TOKEN"]
