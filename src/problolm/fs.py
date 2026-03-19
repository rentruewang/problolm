# Copyright (c) ProBloLM Authors - All Rights Reserved

"The file system that supports navigation to past history."

import abc
import dataclasses as dcls
import functools
import logging
import typing
from abc import ABC
from typing import NamedTuple, Self

from git import Blob, Submodule, Tree
from rich.tree import Tree as RichTree

__all__ = ["TrieNode"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class TrieNode(ABC):
    path: str

    @abc.abstractmethod
    def children(self) -> list[TrieNode]: ...


@dcls.dataclass(frozen=True)
class Folder(TrieNode):
    """
    The folder node in the file system trie.
    """

    items: dict[str, TrieNode] = dcls.field(default_factory=dict)

    def __rich__(self):

        def add_tree(node: TrieNode, tree: RichTree):

            for children in node.children():
                subtree = tree.add(label=children.path)
                add_tree(children, subtree)

        tree = RichTree(label=self.path)
        add_tree(self, tree)
        return tree

    def __truediv__(self, key: str) -> TrieNode:
        return self.__go_one_level_below(key)

    def __go_one_level_below(self):
        return self.items[key]

    def add_folder(self, key: str) -> Self:
        new_node = type(self)(key)
        self.items[key] = new_node
        return new_node

    def add_file(self, key: str, lines: bytes) -> File:
        new_node = File(path=key, data=lines)
        self.items[key] = new_node
        return new_node

    @typing.override
    def children(self) -> list[TrieNode]:
        return list(self.items.values())


@dcls.dataclass(frozen=True)
class File(TrieNode):
    data: bytes

    def read(self) -> str:
        """
        Try to read the body as text, if failed, raise `UnicodeDecodeError`.

        Returns:
            The string if the decoding succeeds.
        """

        content = self._utf8_content

        if isinstance(content, str):
            return content

        raise UnicodeDecodeError(
            "utf-8", self.data, 0, len(self.data), "Invalid UTF-8 content"
        )

    @functools.cached_property
    def _utf8_content(self):
        try:
            return self.data.decode("utf-8")
        except Exception:
            return None

    @typing.override
    def children(self) -> list[TrieNode]:
        return []


def consume(obj: Tree, /) -> Folder:
    """
    Consume the object (must be a git tree), and produce the folder structure.

    Args:
        obj: The git tree object.

    Returns:
        The root folder for the given file system.
    """

    root = Folder("")

    if not isinstance(obj, Tree):
        raise TypeError(f"Expected a git tree, got {obj}.")

    for file in obj.traverse():
        _handle_traversal(file=file, root=root)

    return root


def _handle_traversal(file, root: Folder):
    match file:
        case Tree():
            handle_tree(file, root=root)
        case Blob():
            handle_blob(file, root=root)
        case Submodule():
            handle_submodule(file, root=root)
        case _:
            raise TypeError(type(file))


def handle_tree(tree: Tree, /, root: Folder) -> TrieNode:
    pp = _create_parent_path(tree, root=root)
    return pp.parent.add_folder(pp.path)


def handle_blob(blob: Blob, /, root: Folder):
    pp = _create_parent_path(blob, root=root)
    return pp.parent.add_file(pp.path, blob.data_stream.read())


def handle_submodule(submodule: Submodule, /, root: Folder):
    raise NotImplementedError


class ParentPath(NamedTuple):
    parent: Folder
    path: str


def _create_parent_path(tree_or_blob: Tree | Blob, root: Folder) -> ParentPath:
    "Create parent folder, and pass the current path (remaining part) back."

    *ancestors, path = str(tree_or_blob.path).split("/")

    node = root

    for folder in ancestors:
        node = node.add_folder(folder)

    return ParentPath(parent=node, path=path)
