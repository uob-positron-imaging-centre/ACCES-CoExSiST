#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_imports.py
# License: GNU v3.0


'''The core package must work without optional analysis dependencies.'''


import  subprocess
import  sys
import  textwrap
from    pathlib  import  Path


archive = Path(__file__).parent / "access_data/access_seed123"


def test_core_without_optional_dependencies():
    code = textwrap.dedent('''
        import importlib.abc
        import sys

        class MissingDependencies(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target = None):
                if fullname.split(".")[0] in {
                    "sklearn", "pyevtk", "tqdm", "astunparse", "liggghts",
                }:
                    raise ModuleNotFoundError(fullname, name = fullname)

        sys.meta_path.insert(0, MissingDependencies())
        import coexist

        assert isinstance(coexist.__version__, str)
        assert "sklearn" not in sys.modules
        parameters = coexist.create_parameters(["a", "b"], [0, 0], [1, 1])
        assert parameters.shape == (2, 4)
        data = coexist.AccessData(sys.argv[1])
        assert len(data.results) > 0

        for analyse in [
            lambda: coexist.sensitivity,
            lambda: __import__("coexist.sensitivity"),
            data.sensitivity,
        ]:
            try:
                analyse()
            except ModuleNotFoundError as error:
                assert error.name == "sklearn"
                assert "coexist[sensitivity]" in str(error)
            else:
                raise AssertionError("Missing dependency was not reported.")

        assert "coexist.sensitivity" not in sys.modules
        assert "sklearn" not in sys.modules
    ''')
    subprocess.run([sys.executable, "-c", code, str(archive)], check = True)
