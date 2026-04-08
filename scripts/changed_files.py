from __future__ import annotations

import os
import subprocess

ZERO_SHA = "0000000000000000000000000000000000000000"


def _git_lines(*args: str) -> list[str]:
    output = subprocess.check_output(["git", *args], text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def changed_python_files(
    *,
    event_name: str,
    before: str | None,
    sha: str | None,
    base_ref: str | None,
    default_branch: str,
) -> list[str]:
    if event_name == "pull_request":
        if not base_ref:
            raise ValueError("base_ref is required for pull_request events")
        files = _git_lines(
            "diff",
            "--diff-filter=AM",
            "--name-only",
            f"origin/{base_ref}...HEAD",
        )
    elif before and before != ZERO_SHA:
        if not sha:
            raise ValueError("sha is required for push events")
        files = _git_lines(
            "diff",
            "--diff-filter=AM",
            "--name-only",
            before,
            sha,
        )
    else:
        merge_base = _git_lines("merge-base", "HEAD", f"origin/{default_branch}")[0]
        files = _git_lines(
            "diff",
            "--diff-filter=AM",
            "--name-only",
            f"{merge_base}...HEAD",
        )

    return [path for path in files if path.endswith(".py")]


def main() -> None:
    event_name = os.environ["GITHUB_EVENT_NAME"]
    before = os.getenv("GITHUB_EVENT_BEFORE")
    sha = os.getenv("GITHUB_SHA")
    base_ref = os.getenv("GITHUB_BASE_REF")
    default_branch = os.getenv("GITHUB_DEFAULT_BRANCH", "main")

    files = changed_python_files(
        event_name=event_name,
        before=before,
        sha=sha,
        base_ref=base_ref,
        default_branch=default_branch,
    )
    print("\n".join(files))


if __name__ == "__main__":
    main()
