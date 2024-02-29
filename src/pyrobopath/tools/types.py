import sys

_version = sys.version_info.minor


if _version < 9:
    from pyrobopath.tools._types_35 import *
else:
    from pyrobopath.tools._types_39 import *
