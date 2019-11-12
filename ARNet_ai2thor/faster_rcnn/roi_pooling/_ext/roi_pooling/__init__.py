import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'faster_rcnn/roi_pooling/_ext/roi_pooling'))

from torch.utils.ffi import _wrap_function
from _roi_pooling import lib as _lib, ffi as _ffi

__all__ = []
def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        locals[symbol] = _wrap_function(fn, _ffi)
        __all__.append(symbol)

_import_symbols(locals())