"""
Variables for working with the a model containing Idle, X(pi/2), Y(pi/2), and Z(pi/2) gates.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from collections import OrderedDict
from pygsti.construction import circuitconstruction as _strc
from pygsti.construction import modelconstruction as _setc

from pygsti.modelpacks._modelpack import SMQModelPack


class _Module(SMQModelPack):
    description = "Idle, X(pi/2), Y(pi/2), Z(pi/2) gates"

    gates = [(), ('Gxpi2', 0), ('Gypi2', 0), ('Gzpi2', 0)]

    _sslbls = [0]

    _germs = [((), ), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gzpi2', 0), ), (('Gxpi2', 0), ('Gzpi2', 0)), (('Gxpi2', 0), ('Gypi2', 0)),
              (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0)), (('Gypi2', 0), ('Gypi2', 0), ('Gzpi2', 0)),
              (('Gxpi2', 0), ('Gypi2', 0), ('Gzpi2', 0)), (('Gxpi2', 0), ('Gypi2', 0), ()), (('Gxpi2', 0), (), ('Gypi2', 0)),
              (('Gxpi2', 0), (), ()), (('Gypi2', 0), (), ()), ((), ('Gxpi2', 0), ('Gzpi2', 0)), ((), ('Gypi2', 0), ('Gzpi2', 0)),
              (('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0), ()), (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0))]

    _germs_lite = [((), ), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gzpi2', 0), ), (('Gxpi2', 0), ('Gzpi2', 0)), (('Gxpi2', 0), ('Gypi2', 0)),
                   (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0)),
                   (('Gypi2', 0), ('Gypi2', 0), ('Gzpi2', 0)), (('Gxpi2', 0), ('Gypi2', 0), ('Gzpi2', 0))]

    _fiducials = [(), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gxpi2', 0), ('Gxpi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0)),
                  (('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0))]

    _prepStrs = [(), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gxpi2', 0), ('Gxpi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0)),
                 (('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0))]

    _effectStrs = [(), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gxpi2', 0), ('Gxpi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0)),
                   (('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0))]

    clifford_compilation = None

    global_fidPairs = [(0, 0), (2, 3), (5, 2), (5, 4)]

    pergerm_fidPairsDict = {
        (('Gxpi2', 0), ): [(1, 1), (3, 4), (4, 2), (5, 5)],
        ((), ): [(0, 3), (1, 1), (5, 5)],
        (('Gzpi2', 0), ): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gypi2', 0), ): [(0, 2), (2, 2), (2, 4), (4, 4)],
        (('Gxpi2', 0), ('Gzpi2', 0)): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gxpi2', 0), ('Gypi2', 0)): [(0, 0), (0, 4), (2, 5), (5, 4)],
        (('Gypi2', 0), (), ()): [(0, 2), (2, 2), (2, 4), (4, 4)],
        (('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0)): [(0, 0), (0, 4), (1, 5), (2, 3), (2, 5), (5, 5)],
        (('Gypi2', 0), ('Gypi2', 0), ('Gzpi2', 0)): [(0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)],
        (('Gxpi2', 0), ('Gypi2', 0), ()): [(0, 0), (0, 4), (2, 5), (5, 4)],
        (('Gxpi2', 0), ('Gypi2', 0), ('Gzpi2', 0)): [(0, 2), (2, 2), (2, 4), (4, 4)],
        (('Gxpi2', 0), (), ('Gypi2', 0)): [(0, 0), (0, 4), (2, 5), (5, 4)],
        (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0)): [(1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)],
        ((), ('Gxpi2', 0), ('Gzpi2', 0)): [(0, 0), (2, 3), (5, 2), (5, 4)],
        ((), ('Gypi2', 0), ('Gzpi2', 0)): [(0, 3), (3, 2), (4, 0), (5, 3)],
        (('Gxpi2', 0), (), ()): [(1, 1), (3, 4), (4, 2), (5, 5)],
        (('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0), ()): [(0, 3), (1, 2), (2, 5), (3, 1), (3, 3), (5, 3)],
        (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gxpi2', 0), ('Gypi2', 0), ('Gypi2', 0)): [(0, 0), (2, 3), (5, 2), (5, 4)]
    }

    global_fidPairs_lite = [(0, 4), (0, 5), (1, 0), (2, 0), (2, 4), (2, 5), (3, 0), (4, 2), (4, 4), (5, 1), (5, 2),
                            (5, 3)]

    pergerm_fidPairsDict_lite = {
        (('Gxpi2', 0), ): [(1, 1), (3, 4), (4, 2), (5, 5)],
        ((), ): [(0, 3), (1, 1), (5, 5)],
        (('Gzpi2', 0), ): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gypi2', 0), ): [(0, 2), (2, 2), (2, 4), (4, 4)],
        (('Gxpi2', 0), ('Gzpi2', 0)): [(0, 0), (2, 3), (5, 2), (5, 4)],
        (('Gxpi2', 0), ('Gypi2', 0)): [(0, 0), (0, 4), (2, 5), (5, 4)],
        (('Gypi2', 0), ('Gypi2', 0), ('Gzpi2', 0)): [(0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)],
        (('Gxpi2', 0), ('Gypi2', 0), ('Gzpi2', 0)): [(0, 2), (2, 2), (2, 4), (4, 4)],
        (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 0)): [(1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)],
        (('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0)): [(0, 0), (0, 4), (1, 5), (2, 3), (2, 5), (5, 5)]
    }

    @property
    def _target_model(self):
        return _setc.build_explicit_model([(0, )], [(), ('Gxpi2', 0), ('Gypi2', 0), ('Gzpi2', 0)],
                                          ['I(0)', 'X(pi/2,0)', 'Y(pi/2,0)', 'Z(pi/2,0)'])


import sys
sys.modules[__name__] = _Module()
