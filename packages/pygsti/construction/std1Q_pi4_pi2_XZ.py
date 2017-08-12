from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Variables for working with the a gate set containing X(pi/4) and Z(pi/2) gates.
"""
from . import gatestringconstruction as _strc
from . import gatesetconstruction as _setc

description = "X(pi/4) and Z(pi/2) gates"

gates = ['Gx','Gz']

prepStrs = _strc.gatestring_list([(),
                                       ('Gx','Gx',),
                                       ('Gx','Gx','Gz'),
                                       ('Gx','Gx','Gx','Gx'),
                                       ('Gx','Gx','Gx','Gx','Gx','Gx'),
                                       ('Gx','Gx','Gz','Gz','Gz')])


effectStrs = _strc.gatestring_list([(),
                                       ('Gx','Gx',),
                                       ('Gz','Gx','Gx'),
                                       ('Gx','Gx','Gx','Gx'),
                                       ('Gx','Gx','Gx','Gx','Gx','Gx'),
                                       ('Gz','Gz','Gz','Gx','Gx')])

germs = _strc.gatestring_list( [('Gx',),
                                ('Gz',),
                                ('Gz','Gz','Gx'),
                                ('Gz','Gz','Gx','Gz','Gx','Gx','Gx'),
                                ('Gz','Gx','Gz','Gz','Gx','Gx','Gx')] )

#Construct a target gateset:  X(pi/4), Z(pi/2)
gs_target = _setc.build_gateset( [2], [('Q0',)],['Gx','Gz'],
                                 ["X(pi/4,Q0)", "Z(pi/2,Q0)"],
                                 prepLabels=["rho0"], prepExpressions=["0"],
                                 effectLabels=["E0"], effectExpressions=["0"],
                                 spamdefs={'0': ('rho0','E0'), '1': ('rho0','remainder') },
                                 basis='pp')
