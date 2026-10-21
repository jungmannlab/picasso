"""Tests for ``picasso._numba_cuda_compat``, the frozen-build redirect of
``numba.cuda`` onto NVIDIA's ``numba_cuda`` target.

None of this needs a GPU: the redirect is pure import machinery. A frozen
build is simulated by setting ``sys.frozen``; ``sys.meta_path`` and
``sys.modules`` are patched per test, so nothing leaks into the session.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from picasso import _numba_cuda_compat as compat

ROOT = Path(__file__).resolve().parent.parent

requires_numba_cuda = pytest.mark.skipif(
    importlib.util.find_spec("numba_cuda") is None,
    reason="numba-cuda is not installed",
)


def _redirectors():
    return [
        f for f in sys.meta_path if type(f).__name__ == "_NumbaCudaRedirector"
    ]


@pytest.fixture
def frozen(monkeypatch):
    """A simulated frozen build, before anything imported ``numba.cuda``."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    for name in [m for m in sys.modules if m == "numba.cuda"]:
        monkeypatch.delitem(sys.modules, name)
    return monkeypatch


class TestInstall:
    def test_no_op_outside_a_frozen_build(self, monkeypatch):
        """A normal install already has numba-cuda's own ``.pth`` redirect."""
        monkeypatch.delattr(sys, "frozen", raising=False)
        before = list(sys.meta_path)
        compat.install()
        assert sys.meta_path == before

    def test_installs_first_in_a_frozen_build(self, frozen):
        frozen.setattr(
            compat, "_numba_cuda_parent_paths", lambda: ["/bundle/numba"]
        )
        compat.install()
        assert type(sys.meta_path[0]).__name__ == "_NumbaCudaRedirector"
        assert sys.meta_path[0]._parent_paths == ["/bundle/numba"]

    def test_installs_only_once(self, frozen):
        frozen.setattr(
            compat, "_numba_cuda_parent_paths", lambda: ["/bundle/numba"]
        )
        compat.install()
        compat.install()
        assert len(_redirectors()) == 1

    def test_too_late_once_numba_cuda_is_imported(self, frozen):
        """Redirecting then would leave two copies of the CUDA target."""
        frozen.setitem(sys.modules, "numba.cuda", object())
        frozen.setattr(
            compat, "_numba_cuda_parent_paths", lambda: ["/bundle/numba"]
        )
        compat.install()
        assert not _redirectors()

    def test_cpu_build_without_numba_cuda(self, frozen):
        frozen.setattr(compat, "_numba_cuda_parent_paths", lambda: None)
        compat.install()
        assert not _redirectors()


class TestParentPaths:
    @requires_numba_cuda
    def test_points_at_the_numba_dir_inside_numba_cuda(self):
        paths = compat._numba_cuda_parent_paths()
        assert paths
        for path in paths:
            assert Path(path).name == "numba"
            assert Path(path).parent.name == "numba_cuda"
            assert (Path(path) / "cuda").is_dir()

    def test_none_when_numba_cuda_is_missing(self, monkeypatch):
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        assert compat._numba_cuda_parent_paths() is None

    def test_none_when_the_lookup_raises(self, monkeypatch, capsys):
        def boom(name):
            raise ImportError("broken bundle")

        monkeypatch.setattr(importlib.util, "find_spec", boom)
        monkeypatch.setenv("PICASSO_DEBUG_CUDA", "1")
        assert compat._numba_cuda_parent_paths() is None
        assert "broken bundle" in capsys.readouterr().err


class TestRedirector:
    def test_ignores_other_modules(self):
        finder = compat._NumbaCudaRedirector(["/nowhere"])
        assert finder.find_spec("numpy") is None
        # a prefix match is not enough
        assert finder.find_spec("numba.cudalike") is None

    def test_defers_to_an_already_imported_module(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "numba.cuda.fake", object())
        finder = compat._NumbaCudaRedirector(["/nowhere"])
        assert finder.find_spec("numba.cuda.fake") is None

    def test_unresolvable_name_is_left_to_the_next_finder(self, frozen):
        finder = compat._NumbaCudaRedirector([os.fspath(ROOT / "nowhere")])
        assert finder.find_spec("numba.cuda") is None

    @requires_numba_cuda
    def test_resolves_numba_cuda_inside_numba_cuda(self, frozen):
        finder = compat._NumbaCudaRedirector(compat._numba_cuda_parent_paths())
        spec = finder.find_spec("numba.cuda")
        assert spec is not None
        assert spec.name == "numba.cuda"
        assert "numba_cuda" in Path(spec.origin).parts

    @requires_numba_cuda
    def test_submodules_follow_the_parent_package_path(self, monkeypatch):
        """Python hands a submodule lookup the parent's ``__path__``, which
        the redirected ``numba.cuda`` spec set to numba_cuda's directory."""
        parents = compat._numba_cuda_parent_paths()
        cuda_dirs = [os.path.join(p, "cuda") for p in parents]
        monkeypatch.delitem(sys.modules, "numba.cuda.cudadrv", raising=False)
        finder = compat._NumbaCudaRedirector(["/nowhere"])
        spec = finder.find_spec("numba.cuda.cudadrv", cuda_dirs)
        assert spec is not None
        assert "numba_cuda" in Path(spec.origin).parts

    def test_debug_log_is_opt_in(self, monkeypatch, capsys):
        monkeypatch.delenv("PICASSO_DEBUG_CUDA", raising=False)
        compat._log("quiet")
        assert capsys.readouterr().err == ""
        monkeypatch.setenv("PICASSO_DEBUG_CUDA", "1")
        compat._log("loud")
        assert "[cuda-compat] loud" in capsys.readouterr().err


@requires_numba_cuda
def test_frozen_import_resolves_the_numba_cuda_target():
    """End to end, in a fresh interpreter: with numba-cuda's own ``.pth``
    finder removed - as in a PyInstaller bundle, which never runs ``.pth``
    files - importing picasso must still make ``numba.cuda`` the numba_cuda
    target."""
    script = (
        "import sys\n"
        "sys.meta_path[:] = [f for f in sys.meta_path\n"
        "                    if type(f).__name__ != 'NumbaCudaFinder']\n"
        "assert 'numba.cuda' not in sys.modules\n"
        "sys.frozen = True\n"
        "import picasso\n"
        "assert type(sys.meta_path[0]).__name__ == '_NumbaCudaRedirector'\n"
        "from numba import cuda\n"
        "print(cuda.__file__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "numba_cuda" in Path(result.stdout.strip()).parts
