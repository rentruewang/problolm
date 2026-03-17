# Copyright (c) ProBloLM Authors - All Rights Reserved

"The file system that supports navigation to past history."

from git import Tree, Blob, Submodule
import dataclasses as dcls
from typing import Self
import dataclasses as dcls
from abc import ABC
import abc

@dcls.dataclass(frozen=True)
class TrieNode(ABC):
    path: str

    @abc.abstractmethod
    def children(self) -> dict[str,TrieNode]:...


@dcls.dataclass(frozen=True)
class Folder(TrieNode):
    path: str
    children: dict[str, TrieNode] = dcls.field(default_factory=dict)

    def __truediv__(self, key: str) -> Self:
        return self.insert(key)

    def insert(self, key: str) -> Self:
        new_node = type(self)(key)
        self.children[key] = new_node
        return new_node

@dcls.dataclass(frozen=True)

ROOT = TrieNode("")


def consume(obj: Tree, /):
    for file in obj.traverse():
        match obj:
            case Tree():
                handle_tree(obj)
            case Blob():
                handle_blob(obj)
            case Submodule():
                handle_submodule(obj)
            case _:
                raise TypeError(type(obj))


def handle_tree(tree: Tree, /):
    paths: list[str] = str(tree.path).split("/")

    node = ROOT
    for folder in paths:
        node = node / folder


def handle_blob(blob: Blob, /):
    pass


def handle_submodule(submodule: Submodule, /):
    raise NotImplementedError
