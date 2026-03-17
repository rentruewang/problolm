# Copyright (c) ProBloLM Authors - All Rights Reserved

"The file system that supports navigation to past history."

import abc
import dataclasses as dcls
import typing
from abc import ABC
from typing import NamedTuple, Self

from git import Blob, Submodule, Tree
from rich.tree import Tree as RichTree

__all__ = ["TrieNode"]


@dcls.dataclass(frozen=True)
class TrieNode(ABC):
    path: str

    @abc.abstractmethod
    def children(self) -> list[TrieNode]: ...


@dcls.dataclass(frozen=True)
class Folder(TrieNode):
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
        return self.items[key]

    def add_folder(self, key: str) -> Self:
        new_node = type(self)(key)
        self.items[key] = new_node
        return new_node

    def add_file(self, key: str, lines: list[str]) -> File:
        new_node = File(path=key, data=lines)
        self.items[key] = new_node
        return new_node

    @typing.override
    def children(self) -> list[TrieNode]:
        return list(self.items.values())


@dcls.dataclass(frozen=True)
class File(TrieNode):
    data: bytes

    @typing.override
    def children(self) -> list[TrieNode]:
        return []


ROOT = Folder("")


def consume(obj: Tree, /):
    for file in obj.traverse():
        match file:
            case Tree():
                handle_tree(file)
            case Blob():
                handle_blob(file)
            case Submodule():
                handle_submodule(file)
            case _:
                raise TypeError(type(file))


def handle_tree(tree: Tree, /) -> TrieNode:
    pp = _create_parent_path(tree)
    return pp.parent.add_folder(pp.path)


def handle_blob(blob: Blob, /):
    pp = _create_parent_path(blob)
    return pp.parent.add_file(pp.path, blob.data_stream.read())


def handle_submodule(submodule: Submodule, /):
    raise NotImplementedError


class ParentPath(NamedTuple):
    parent: Folder
    path: str


def _create_parent_path(tb: Tree | Blob) -> ParentPath:
    *ancestors, path = str(tb.path).split("/")

    node = ROOT
    for folder in ancestors:
        node = node.add_folder(folder)

    return ParentPath(parent=node, path=path)
