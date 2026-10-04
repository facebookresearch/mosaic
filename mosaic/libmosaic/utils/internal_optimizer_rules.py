# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Extension hook for additional OPTIMIZER categorization rules.

The constants exported here are empty by default. Downstream builds may
override this module to inject extra optimizer filename substrings or
frame names without forking ``data_utils.py``.
"""

INTERNAL_OPTIMIZER_FILENAME_SUBSTRINGS: tuple = ()
INTERNAL_OPTIMIZER_NAMES: frozenset = frozenset()
