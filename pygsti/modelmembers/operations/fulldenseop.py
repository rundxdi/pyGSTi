"""
The FullDenseOp class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from .linearop import LinearOperator as _LinearOperator
from .denseop import DenseOperator as _DenseOperator


class FullDenseOp(_DenseOperator):
    """
    An operation matrix that is fully parameterized.

    That is, each element of the operation matrix is an independent parameter.

    Parameters
    ----------
    m : array_like or LinearOperator
        a square 2D array-like or LinearOperator object representing the operation action.
        The shape of m sets the dimension of the operation.

    evotype : {"statevec", "densitymx", "auto"}
        The evolution type.  If `"auto"`, then `"statevec"` is used if and only if `m`
        has a complex datatype.
    """

    def __init__(self, m, evotype="densitymx"):
        """
        Initialize a FullDenseOp object.

        Parameters
        ----------
        m : array_like or LinearOperator
            a square 2D array-like or LinearOperator object representing the operation action.
            The shape of m sets the dimension of the operation.

        evotype : {"statevec", "densitymx", "auto"}
            The evolution type.  If `"auto"`, then `"statevec"` is used if and only if `m`
            has a complex datatype.
        """
        m = _LinearOperator.convert_to_matrix(m)
        _DenseOperator.__init__(self, m, evotype)

        d = self.dim
        self._paramlbls = _np.array(["MxElement %d,%d" % (i, j) for i in range(d) for j in range(d)],
                                    dtype=object)

    def set_dense(self, m):
        """
        Set the dense-matrix value of this operation.

        Attempts to modify operation parameters so that the specified raw
        operation matrix becomes mx.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        m : array_like or LinearOperator
            An array of shape (dim, dim) or LinearOperator representing the operation action.

        Returns
        -------
        None
        """
        mx = _LinearOperator.convert_to_matrix(m)
        if(mx.shape != (self.dim, self.dim)):
            raise ValueError("Argument must be a (%d,%d) matrix!"
                             % (self.dim, self.dim))
        self.base[:, :] = _np.array(mx)
        self.dirty = True

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.size

    def to_vector(self):
        """
        Get the operation parameters as an array of values.

        Returns
        -------
        numpy array
            The operation parameters as a 1D array with length num_params().
        """
        return self.base.flatten()

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the operation using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of operation parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this operation's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        dirty_value : bool, optional
            The value to set this object's "dirty flag" to before exiting this
            call.  This is passed as an argument so it can be updated *recursively*.
            Leave this set to `True` unless you know what you're doing.

        Returns
        -------
        None
        """
        assert(self.base.shape == (self.dim, self.dim))
        self.base[:, :] = v.reshape((self.dim, self.dim))
        self.dirty = dirty_value

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this operation.

        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single operation parameter.  Thus, each column is of length
        op_dim^2 and there is one column per operation parameter.

        Parameters
        ----------
        wrt_filter : list or numpy.ndarray
            List of parameter indices to take derivative with respect to.
            (None means to use all the this operation's parameters.)

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        derivMx = _np.identity(self.dim**2, self.base.dtype)

        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Whether this operation has a non-zero Hessian with respect to its parameters.

        (i.e. whether it only depends linearly on its parameters or not)

        Returns
        -------
        bool
        """
        return False
