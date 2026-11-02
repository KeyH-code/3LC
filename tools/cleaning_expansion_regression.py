"""Run synthetic regression with legacy scratch paths isolated in this project.

Only in-memory test module constants change; frozen old source files stay intact.
No model execution and no reading old host scratch/PREVIOUS directories.
"""
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='review-regression-', dir=ROOT) as temp:
        scratch = Path(temp)
        os.environ['PI_SCRATCH_DIR'] = str(scratch)
        suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
        modules = list(sys.modules.values())
        modules += [value for module in list(modules) if isinstance(module, types.ModuleType)
                    for value in vars(module).copy().values() if isinstance(value, types.ModuleType)]
        for module in modules:
            filename = getattr(module, '__file__', None)
            if not filename or not Path(filename).resolve().is_relative_to(ROOT):
                continue
            if 'SCRATCH' in vars(module): module.SCRATCH = scratch
            if 'PREVIOUS' in vars(module): module.PREVIOUS = scratch/'no-previous-attempt'
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        return 0 if result.wasSuccessful() else 1


if __name__ == '__main__': sys.exit(main())
