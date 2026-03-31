# Copyright (c) ProBloLM Authors - All Rights Reserved

"The file system that supports navigation to past history."

import abc
import dataclasses as dcls
import functools
import logging
import typing
from collections import abc as cabc

import git
from rich import tree

if typing.TYPE_CHECKING:
    from . import commits

__all__ = ["TrieNode", "Folder", "File"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class TrieNode(abc.ABC):
    class Visitor[T](abc.ABC):
        def visit(self, node: TrieNode) -> T:
            return node.accept(self)

        @abc.abstractmethod
        def visit_folder(self, folder: Folder, /) -> T: ...

        @abc.abstractmethod
        def visit_file(self, file: File, /) -> T: ...

    commit: commits.Commit
    path: str
    parent: TrieNode | None

    @property
    def is_root(self):
        return self.parent is None

    def full_path(self) -> str:
        """
        Get the full path for the node.

        Returns:
            The path with "/" as separator.
        """

        reversed_paths: list[str] = []

        node: TrieNode | None = self

        while node:
            reversed_paths.append(node.path)
            node = node.parent

        return "/".join(reversed(reversed_paths))

    @abc.abstractmethod
    def accept[T](self, visitor: Visitor[T]) -> T: ...

    @abc.abstractmethod
    def children(self) -> list[TrieNode]: ...


@dcls.dataclass(frozen=True)
class Folder(TrieNode):
    """
    The folder node in the file system trie.
    """

    items: dict[str, TrieNode] = dcls.field(default_factory=dict)

    def __contains__(self, obj: str) -> bool:
        return obj in self.items.keys()

    def __rich__(self):

        def add_tree(node: TrieNode, t: tree.Tree):

            for children in node.children():
                subtree = t.add(label=children.path)
                add_tree(children, subtree)

        t = tree.Tree(label=self.path)
        add_tree(self, t)
        return t

    def __truediv__(self, key: str) -> TrieNode:
        paths = key.split("/")

        node = self

        for path in paths:
            if not isinstance(node, Folder):
                raise ValueError(f"Key: {key} is not valid.")

            node = node.__go_one_level_below(path)

        return node

    def list_files(self) -> cabc.Generator[File]:
        """
        List the files in the sub-directory.

        Yields:
            cabc.Generator[File]: _description_
        """

        class ListFileVisitor(TrieNode.Visitor[cabc.Generator[File]]):

            def visit_file(self, file: File) -> cabc.Generator[File, None, None]:
                yield file

            def visit_folder(self, folder: Folder) -> cabc.Generator[File, None, None]:
                yield from folder.list_files()

        lister = ListFileVisitor()

        for child in self.items.values():
            yield from lister.visit(child)

    def __go_one_level_below(self, key: str):
        try:
            return self.items[key]
        except KeyError:
            raise ValueError(
                f"{self.full_path()}/{key} is not present in {self.commit}."
            )

    def guarded_get[T](self, key: str, typ: type[T]) -> T | None:
        """
        Do a `get`, but with type guarding.
        """

        if key not in self:
            return None

        result = self.items[key]
        assert isinstance(result, typ)
        assert issubclass(typ, TrieNode)
        return result

    def add_folder(self, key: str) -> Folder:
        """
        Add a folder to the current folder.
        If present, fetch the existing folder.
        """

        if folder := self.guarded_get(key, Folder):
            return folder

        new_node = Folder(commit=self.commit, path=key, parent=self)
        self.items[key] = new_node
        return new_node

    def add_file(self, key: str, lines: bytes) -> File:
        """
        Add a file to the current folder.
        If present, fetch the existing file.
        """

        if file := self.guarded_get(key, File):
            return file

        new_node = File(commit=self.commit, path=key, data=lines, parent=self)
        self.items[key] = new_node
        return new_node

    def rm(self, key: str):
        item = self.items[key]
        del self.items[key]
        return item

    @typing.override
    def accept[T](self, visitor: TrieNode.Visitor[T]) -> T:
        return visitor.visit_folder(self)

    @typing.override
    def children(self) -> list[TrieNode]:
        return list(self.items.values())

    @classmethod
    def init_root_for(cls, commit: commits.Commit) -> typing.Self:
        return cls(commit=commit, path="", parent=None)


@dcls.dataclass(frozen=True)
class File(TrieNode):
    "The file nodes in the file system tree."

    data: bytes
    "The data stored in an object. Might be decodable to a string."

    def __repr__(self) -> str:
        full_path = self.full_path()

        if self.is_binary():
            return f"Binary({full_path})"
        else:
            lines = self.read().count("\n") + 1
            return f"Text({full_path}, lines={lines})"

    def is_binary(self) -> bool:
        return self._utf8_content is None

    def is_text(self) -> bool:
        return not self.is_binary()

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
    def accept[T](self, visitor: TrieNode.Visitor[T]) -> T:
        return visitor.visit_file(self)

    @typing.override
    def children(self) -> list[TrieNode]:
        return []


def consume(commit: commits.Commit, /) -> Folder:
    """
    Consume the object (must be a git tree), and produce the folder structure.

    Args:
        commit: The commit object.

    Returns:
        The root folder for the given file system.
    """

    obj = commit.git.tree
    root = Folder.init_root_for(commit)

    if not isinstance(obj, git.Tree):
        raise TypeError(f"Expected a git tree, got {obj}.")

    for file in obj.traverse():
        _handle_traversal(file=file, root=root)

    return root


def _handle_traversal(file, root: Folder) -> None:
    match file:
        case git.Tree():
            _handle_tree(file, root=root)
        case git.Blob():
            _handle_blob(file, root=root)
        case git.Submodule():
            _handle_submodule(file, root=root)
        case _:
            raise TypeError(type(file))


def _handle_tree(tree: git.Tree, /, root: Folder) -> Folder:
    LOGGER.debug("Recurse into sub-tree: %s", tree.path)
    pp = _create_parent_path(tree, root=root)
    folder = pp.parent.add_folder(pp.path)
    return folder


def _handle_blob(blob: git.Blob, /, root: Folder) -> File:
    LOGGER.debug("Recurse into file: %s", blob.path)
    pp = _create_parent_path(blob, root=root)
    return pp.parent.add_file(pp.path, blob.data_stream.read())


def _handle_submodule(submodule: git.Submodule, /, root: Folder) -> typing.NoReturn:
    raise NotImplementedError


class ParentPath(typing.NamedTuple):
    parent: Folder
    path: str


def _create_parent_path(tree_or_blob: git.Tree | git.Blob, root: Folder) -> ParentPath:
    "Create parent folder, and pass the current path (remaining part) back."

    *ancestors, path = str(tree_or_blob.path).split("/")

    node = root

    for folder in ancestors:
        node = node.add_folder(folder)

    return ParentPath(parent=node, path=path)
