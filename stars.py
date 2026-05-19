# Copyright (c) ProBloLM Authors - All Rights Reserved

from rich import progress
from .consts import github
import pandas as pd
from github3 import exceptions as gh_exc


def get_all_user_stars(username: str):
    print(f"Fetching all stars for user: {username}...")

    try:
        # Use starred_by directly on the GitHub object for better compatibility
        starred_repos = github().starred_by(username)

        count = 0
        print(f"{'#':<4} | {'Repository Name':<40} | {'Stars'}")
        print("-" * 60)

        answer = []

        for repo in progress.track(starred_repos):
            count += 1

            # Use getattr to safely fetch attributes in case they are missing
            # Some versions of the library require repo.repository.full_name
            # if the object is a 'StarredRepository' wrapper.
            full_name = getattr(repo, "full_name", "Unknown/Private")
            stars = getattr(repo, "stargazers_count", 0)
            answer.append(
                {
                    "full_name": full_name,
                    "count": count,
                    "stars": stars,
                }
            )
            print(f"{count:<4} | {full_name:<40} | {stars}")

        print("-" * 60)
        print(f"Total stars found: {count}")
        return pd.DataFrame(answer)
    except gh_exc.ForbiddenError:
        print("\nRate limit hit! Anonymous requests are limited to 60 per hour.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
