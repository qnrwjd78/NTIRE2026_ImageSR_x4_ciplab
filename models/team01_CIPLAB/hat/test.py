from __future__ import annotations

import os.path as osp

from basicsr.test import test_pipeline

from .archs import HAT  # noqa: F401
from .models import HATModel  # noqa: F401


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
