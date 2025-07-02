import sys

_version = sys.version_info.minor


if _version < 9:
    from ._types_35 import *
else:
    from ._types_39 import *
