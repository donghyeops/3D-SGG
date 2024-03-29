# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# TODO: make this fold self-contained, only depends on utils package

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'faster_rcnn/datasets'))

## NOTE: obsolete
import os.path as osp

# http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
def _which(program):
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None
"""
if _which(MATLAB) is None:
    msg = ("MATLAB command '{}' not found. "
           "Please add '{}' to your PATH.").format(MATLAB, MATLAB)
    raise EnvironmentError(msg)
"""
