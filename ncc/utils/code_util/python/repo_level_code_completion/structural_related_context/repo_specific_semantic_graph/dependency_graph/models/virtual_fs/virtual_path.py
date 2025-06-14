import os
import pathlib
import sys

import fs.osfs
import fs.path
import fs.permissions
from fs.base import FS
from fs.enums import ResourceType


class VirtualPath(pathlib.PosixPath):
    """
    Inspired by https://github.com/smheidrich/pathlib-fs
    A pathlib-like object that uses a PyFilesystem as its backend.
    """

    if sys.version_info < (3, 12):

        def __new__(cls, fs: FS, *pathsegments, disallow_str=False):
            self = super().__new__(cls, *pathsegments)
            self.fs = fs
            self.disallow_str = disallow_str
            return self

    else:

        def __init__(self, fs: FS, *pathsegments, disallow_str=False):
            super().__init__(*pathsegments)
            self.fs = fs
            self.disallow_str = disallow_str

    @classmethod
    def home(cls, disallow_str=False):
        # we need an explicit reference here because super().home() would try to
        # instantiate using cls, which is our current class -> doesn't work
        home_parts = pathlib.Path.home().parts
        root = home_parts[0]
        rest = home_parts[1:] if len(home_parts) > 1 else ""
        return cls(FS(root), *rest, disallow_str=disallow_str)

    @classmethod
    def cwd(cls, disallow_str=False):
        # cf. comment on home() above
        cwd_parts = pathlib.Path.cwd().parts
        root = cwd_parts[0]
        rest = cwd_parts[1:] if len(cwd_parts) > 1 else ""
        return cls(FS(root), *rest, disallow_str=disallow_str)

    # methods that emulate pathlib via PyFilesystem

    def open(self, *args, **kwargs):
        return self.fs.open(self.relative_fs_path, *args, **kwargs)

    def touch(self):
        self.fs.touch(self.relative_fs_path)

    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        permissions = fs.permissions.Permissions(mode=mode)
        if parents and self.parts and not self.parent.is_dir():
            # it's fine to only progress up to where the PyFilesystem part of the
            # path starts, as the former is guaranteed to exist (at least for OSFS),
            # so we never have to create it
            self.parent.mkdir(mode=mode, parents=parents, exist_ok=True)
        self.fs.makedir(
            self.relative_fs_path, permissions=permissions, recreate=exist_ok
        )

    def rename(self, target):
        if isinstance(target, VirtualPath):
            if target.fs != self.fs:
                raise ValueError(
                    "rename is only supported for {} objects based on "
                    "the same PyFilesystem object".format(self.__class__.__name__)
                )
            target = target.relative_fs_path
        else:
            raise NotImplementedError("")
            # what should happen here:
            # - relative paths are interpreted as relative to cwd by pathlib => only
            #   makes sense for isinstance(fs, OsFs)
            # - absolute paths need some method of being interpreted with respect to
            #   FS (probably also only makes sense for OsFs, doesn't it?)
        self.fs.move(self.relative_fs_path, target, overwrite=False)

    def rmdir(self):
        self.fs.removedir(self.relative_fs_path)

    def unlink(self):
        self.fs.remove(self.relative_fs_path)

    def chmod(self, mode):
        raise NotImplementedError(
            "PyFilesystem has no chmod() equivalent that I "
            "know of, so {}.chmod won't work either".format(self.__class__.__name__)
        )

    def symlink_to(self, other):
        raise NotImplementedError(
            "how does one create symlinks in PyFilesystem? "
            "let me know at https://github.com/smheidrich/pathlib-fs/issues/new"
        )

    def exists(self):
        try:
            return self.fs.exists(self.relative_fs_path)
        except Exception:
            return False

    def is_block_device(self):
        if not self.exists():
            return False
        return self.fs.gettype(self.relative_fs_path) == ResourceType.block_special_file

    def is_char_device(self):
        if not self.exists():
            return False
        return self.fs.gettype(self.relative_fs_path) == ResourceType.character

    def is_dir(self):
        if not self.exists():
            return False
        return self.fs.isdir(self.relative_fs_path)

    def is_file(self):
        if not self.exists():
            return False
        return self.fs.isfile(self.relative_fs_path)

    def is_fifo(self):
        if not self.exists():
            return False
        return self.fs.gettype(self.relative_fs_path) == ResourceType.fifo

    def is_socket(self):
        if not self.exists():
            return False
        return self.fs.gettype(self.relative_fs_path) == ResourceType.socket

    def is_symlink(self):
        if not self.exists():
            return False
        # TODO report this to PyFilesystem... islink should just return False if
        # the path doesn't exist, because isfile and isdir do the same; but right
        # now it raises an exception, so we have to do this roundabout thing
        return self.exists() and self.fs.islink(self.relative_fs_path)

    def iterdir(self):
        for d in self.fs.listdir(self.relative_fs_path):
            yield self.__class__(
                self.fs,
                fs.path.join(self.relative_fs_path, d),
                disallow_str=self.disallow_str,
            )

    def glob(self, pattern):
        for match in self.fs.glob(fs.path.join(self.relative_fs_path, pattern)):
            yield self.__class__(self.fs, match.path, disallow_str=self.disallow_str)

    def rglob(self, pattern):
        for match in self.fs.glob(fs.path.join(self.relative_fs_path, "**", pattern)):
            yield self.__class__(self.fs, match.path, disallow_str=self.disallow_str)

    def __eq__(self, other):
        if not isinstance(other, VirtualPath):
            return NotImplemented
        return super().__eq__(other) and self.fs == other.fs

    def owner(self):
        info = self.fs.getinfo(self.relative_fs_path, namespaces=["access"])
        if info.has_namespace("access"):
            return info.user
        else:
            return None

    def group(self):
        info = self.fs.getinfo(self.relative_fs_path, namespaces=["access"])
        if info.has_namespace("access"):
            return info.group
        else:
            return None

    def stat(self):
        fields = [
            "mode",
            "ino",
            "dev",
            "nlink",
            "uid",
            "gid",
            "size",
            "atime",
            "mtime",
            "ctime",
        ]
        # Seems most other filesystems except OSFS will not support stat namespace
        info = self.fs.getinfo(self.relative_fs_path, namespaces=["stat"])
        if info.has_namespace("stat"):
            return os.stat_result([info.raw["stat"]["st_" + x] for x in fields])
        else:
            return None

    def lstat(self):
        fields = [
            "mode",
            "ino",
            "dev",
            "nlink",
            "uid",
            "gid",
            "size",
            "atime",
            "mtime",
            "ctime",
        ]
        # Seems most other filesystems except OSFS will not support lstat namespace
        info = self.fs.getinfo(self.relative_fs_path, namespaces=["lstat"])
        if info.has_namespace("lstat"):
            return os.stat_result([info.raw["lstat"]["st_" + x] for x in fields])
        else:
            return None

    # various "representations"

    @property
    def relative_fs_path(self) -> str:
        """
        The path relative to ``fs``, i.e. what PyFilesystem considers a path
        """
        return super().__str__()

    def as_pathlib_path(self):
        # this assumes that pathlib.Path -> Posix/Windows resolution will match how
        # PyFilesystem determines OS flavor... is this the case? not sure.
        # return pathlib.Path(self.fs.getospath(self.relative_fs_path))
        # TODO ^ this should work but getospath() returns bytes instead of str...
        # TODO --> report this as a bug!!
        return pathlib.Path(self.relative_fs_path)

    def as_posix(self):
        return self.as_pathlib_path().as_posix()

    def as_str(self):
        return str(self.as_pathlib_path())

    def __str__(self):
        if not self.disallow_str:
            return self.as_str()
        else:
            raise ValueError(
                "str() not allowed for this {} instance".format(self.__class__)
            )

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.fs, repr(super().__str__())
        )

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(self.relative_fs_path) ^ hash(self.fs)
            return self._hash

    def as_uri(self):
        return self.fs.geturl(self.relative_fs_path)

    # stuff that can just wrap pathlib methods directly
    # TODO I probably have to overwrite just one internal method to make all of
    # these work without having to write them here...

    def joinpath(self, *other):
        p = super().joinpath(*other)
        return self.__class__(self.fs, *(p.parts), disallow_str=self.disallow_str)

    def __truediv__(self, x):
        p = super().__truediv__(x)
        return self.__class__(self.fs, *(p.parts), disallow_str=self.disallow_str)

    def with_segments(self, *pathsegments):
        """Construct a new path object from any number of path-like objects.
        Subclasses may override this method to customize how new path objects
        are created from methods like `iterdir()`.
        """
        return type(self)(self.fs, *pathsegments)

    def with_name(self, name):
        p = super().with_name(name)
        return self.__class__(self.fs, *(p.parts), disallow_str=self.disallow_str)

    def with_suffix(self, suffix):
        p = super().with_suffix(suffix)
        return self.__class__(self.fs, *(p.parts), disallow_str=self.disallow_str)

    @property
    def parent(self):
        p = super().parent
        return self.__class__(self.fs, *(p.parts), disallow_str=self.disallow_str)

    @property
    def parents(self):
        if len(self.parts) == 1:
            return [self.parent]
        else:
            return [self.parent] + self.parent.parents

    def relative_to(self, other):
        if isinstance(other, VirtualPath):
            if other.fs != self.fs:
                raise ValueError(
                    "relative_to is only supported for {} objects based "
                    "on the same PyFilesystem object".format(self.__class__.__name__)
                )
        else:
            other = self.with_segments(other)
        return self.__class__(self.fs, *(super().relative_to(other).parts))

    def expanduser(self):
        # do nothing, as this can't be implemented in a general fashion
        return self

    def absolute(self):
        # FS objects have no concept of a current directory
        return self.with_segments(fs.path.abspath(self.relative_fs_path))

    def resolve(self, strict: bool = False):
        abs_path = self.absolute()
        resolved_path = pathlib.Path(str(abs_path)).resolve(strict=False)
        p = self.__class__(self.fs, *resolved_path.parts)
        if strict and not p.exists():
            raise FileNotFoundError(p)
        return p

    if sys.version_info < (3, 9):

        def is_relative_to(self, *other):
            try:
                self.relative_to(*other)
                return True
            except ValueError:
                return False
