
import numpy as _np
import scipy.sparse as _sps
from .linearop import LinearOperator as _LinearOperator
from ...evotypes import Evotype as _Evotype


class RepeatedOp(_LinearOperator):
    """
    An operation map that is the composition of a number of map-like factors (possibly other `LinearOperator`s)

    Parameters
    ----------
    op_to_repeat : list
        A `LinearOperator`-derived object that is repeated
        some integer number of times to produce this operator.

    num_repetitions : int
        the power to exponentiate `op_to_exponentiate` to.

    evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
        the evolution type.  `"auto"` uses the evolution type of
        `op_to_repeat`.
    """

    def __init__(self, op_to_repeat, num_repetitions, evotype="auto"):
        #We may not actually need to save these, since they can be inferred easily
        self.repeated_op = op_to_repeat
        self.num_repetitions = num_repetitions

        dim = op_to_repeat.dim

        if evotype == "auto":
            evotype = op_to_repeat._evotype
        evotype = _Evotype.cast(evotype)
        rep = evotype.create_repeated_rep(self.repeated_op._rep, self.num_repetitions, dim)
        _LinearOperator.__init__(self, rep, evotype)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.repeated_op]

    def set_time(self, t):
        """
        Sets the current time for a time-dependent operator.

        For time-independent operators (the default), this function does nothing.

        Parameters
        ----------
        t : float
            The current time.

        Returns
        -------
        None
        """
        self.repeated_op.set_time(t)

    def copy(self, parent=None, memo=None):
        """
        Copy this object.

        Parameters
        ----------
        parent : Model, optional
            The parent model to set for the copy.

        Returns
        -------
        LinearOperator
            A copy of this object.
        """
        # We need to override this method so that factor operations have their
        # parent reset correctly.
        if memo is not None and id(self) in memo: return memo[id(self)]
        cls = self.__class__  # so that this method works for derived classes too
        copyOfMe = cls(self.repeated_op.copy(parent, memo), self.num_repetitions, self._evotype)
        return self._copy_gpindices(copyOfMe, parent, memo)

    def to_sparse(self):
        """
        Return the operation as a sparse matrix

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        if self.num_repetitions == 0:
            return _sps.identity(self.dim, dtype=_np.dtype('d'), format='csr')

        op = self.repeated_op.to_sparse()
        mx = op.copy()
        for i in range(self.num_repetitions - 1):
            mx = mx.dot(op)
        return mx

    def to_dense(self):
        """
        Return this operation as a dense matrix.

        Returns
        -------
        numpy.ndarray
        """
        op = self.repeated_op.to_dense()
        return _np.linalg.matrix_power(op, self.num_repetitions)

    #def torep(self):
    #    """
    #    Return a "representation" object for this operation.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    OpRep
    #    """
    #    if self._evotype == "densitymx":
    #        return replib.DMOpRepExponentiated(self.repeated_op.torep(), self.power, self.dim)
    #    elif self._evotype == "statevec":
    #        return replib.SVOpRepExponentiated(self.repeated_op.torep(), self.power, self.dim)
    #    elif self._evotype == "stabilizer":
    #        nQubits = int(round(_np.log2(self.dim)))  # "stabilizer" is a unitary-evolution type mode
    #        return replib.SVOpRepExponentiated(self.repeated_op.torep(), self.power, nQubits)
    #    assert(False), "Invalid internal _evotype: %s" % self._evotype

    #FUTURE: term-related functions (maybe base off of ComposedOp or use a composedop to generate them?)
    # e.g. ComposedOp([self.repeated_op] * power, dim, evotype)

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        return self.repeated_op.paramter_labels

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.repeated_op.num_params

    def to_vector(self):
        """
        Get the operation parameters as an array of values.

        Returns
        -------
        numpy array
            The operation parameters as a 1D array with length num_params().
        """
        return self.repeated_op.to_vector()

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
        assert(len(v) == self.num_params)
        self.repeated_op.from_vector(v, close, dirty_value)
        self.dirty = dirty_value

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this operation.

        Constructs a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single operation parameter.  Thus, each column is of length
        op_dim^2 and there is one column per operation parameter. An
        empty 2D array in the StaticDenseOp case (num_params == 0).

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
        mx = self.repeated_op.to_dense()

        mx_powers = {0: _np.identity(self.dim, 'd'), 1: mx}
        for i in range(2, self.num_repetitions):
            mx_powers[i] = _np.dot(mx_powers[i - 1], mx)

        dmx = _np.transpose(self.repeated_op.deriv_wrt_params(wrt_filter))  # (num_params, dim^2)
        dmx.shape = (dmx.shape[0], self.dim, self.dim)  # set shape for multiplication below

        deriv = _np.zeros((self.dim, dmx.shape[0], self.dim), 'd')
        for k in range(1, self.num_repetitions + 1):
            #deriv += mx_powers[k-1] * dmx * mx_powers[self.num_repetitions-k]
            deriv += _np.dot(mx_powers[k - 1], _np.dot(dmx, mx_powers[self.num_repetitions - k]))
            #        (D,D) * ((P,D,D) * (D,D)) => (D,D) * (P,D,D) => (D,P,D)

        deriv = _np.moveaxis(deriv, 1, 2)
        deriv = deriv.reshape((self.dim**2, deriv.shape[2]))
        return deriv

    def __str__(self):
        """ Return string representation """
        s = "Repeated operation that repeates the below op %d times\n" % self.num_repetitions
        s += str(self.repeated_op)
        return s
