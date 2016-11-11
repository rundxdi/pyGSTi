from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for working with MPI processor distributions"""

import numpy as _np

def distribute_indices(indices, comm, allow_split_comm=True):
    """ 
    Partition an array of indices (any type) evenly among `comm`'s processors.

    Parameters
    ----------
    indices : list
        An array of items (any type) which are to be partitioned.

    comm : mpi4py.MPI.Comm
        The communicator which specifies the number of processors and
        which may be split into returned sub-communicators.

    allow_split_comm : bool
        If True, when there are more processors than indices, 
        multiple processors will be given the *same* set of local
        indices and `comm` will be split into sub-communicators,
        one for each group of processors that are given the same
        indices.  If False, then "extra" processors are simply given
        nothing to do, i.e. empty lists of local indices.

    Returns
    -------
    loc_indices : list
        A list containing the elements of `indices` belonging to the current
        processor.

    owners : dict
        A dictionary mapping the elements of `indices` to integer ranks, such
        that `owners[el]` gives the rank of the processor responsible for 
        communicating that element's results to the other processors.  Note that
        in the case when `allow_split_comm=True` and multiple procesors have
        computed the results for a given element, only a single (the first) 
        processor rank "owns" the element, and is thus responsible for sharing
        the results.  This notion of ownership is useful when gathering the
        results.
        
    loc_comm : mpi4py.MPI.Comm or None
        The local communicator for the group of processors which have been
        given the same `loc_indices` to compute, obtained by splitting `comm`.
        If `loc_indices` is unique to the current processor, or if 
        `allow_split_comm` is False, None is returned.
    """
    if comm is None:
        nprocs, rank = 1, 0
    else:
        nprocs = comm.Get_size()
        rank = comm.Get_rank()

    loc_indices, owners = distribute_indices_base(indices, nprocs, rank,
                                                  allow_split_comm)

    #Split comm into sub-comms when there are more procs than
    # indices, resulting in all procs getting only a 
    # single index and multiple procs getting the *same*
    # (single) index.
    if nprocs > len(indices) and (comm is not None) and allow_split_comm:
        loc_comm = comm.Split(color=loc_indices[0], key=rank)  
    else: 
        loc_comm = None

    return loc_indices, owners, loc_comm

def distribute_indices_base(indices, nprocs, rank, allow_split_comm=True):
    """ 
    Partition an array of "indices" evenly among a given number of "processors"

    This function is similar to :func:`distribute_indices`, but allows for more
    a more generalized notion of what a "processor" is, since the number of
    processors and rank are given independently and do not have to be
    associated with an MPI comm.  Note also that `indices` can be an arbitrary
    list of items, making this function very general.

    Parameters
    ----------
    indices : list
        An array of items (any type) which are to be partitioned.
       
    nprocs : int
        The number of "processors" to distribute the elements of
        `indices` among.

    rank : int
        The rank of the current "processor" (must be an integer
        between 0 and `nprocs-1`).  Note that this value is not
        obtained from any MPI communicator.

    allow_split_comm : bool
        If True, when there are more processors than indices, 
        multiple processors will be given the *same* set of local
        indices.  If False, then extra processors are simply given
        nothing to do, i.e. empty lists of local indices.

    Returns
    -------
    loc_indices : list
        A list containing the elements of `indices` belonging to the current
        processor (i.e. the one specified by `rank`).

    owners : dict
        A dictionary mapping the elements of `indices` to integer ranks, such
        that `owners[el]` gives the rank of the processor responsible for 
        communicating that element's results to the other processors.  Note that
        in the case when `allow_split_comm=True` and multiple procesors have
        computed the results for a given element, only a single (the first) 
        processor rank "owns" the element, and is thus responsible for sharing
        the results.  This notion of ownership is useful when gathering the
        results.
    """
    nIndices = len(indices)
    if nIndices == 0: # special case when == 0
        return [], {}

    if nprocs >= nIndices:
        if allow_split_comm:
            nloc_std =  nprocs // nIndices
            extra = nprocs - nloc_std*nIndices # extra procs
            if rank < extra*(nloc_std+1):
                loc_indices = [ indices[rank // (nloc_std+1)] ]
            else:
                loc_indices = [ indices[
                        extra + (rank-extra*(nloc_std+1)) // nloc_std ] ]
    
            # owners dict gives rank of first (chief) processor for each index
            # (the "owner" of a given index is responsible for communicating
            #  results for that index to the other processors)
            owners = { indices[i]: i*(nloc_std+1) for i in range(extra) }
            owners.update( { indices[i]: extra*(nloc_std+1) + (i-extra)*nloc_std
                             for i in range(extra, nIndices) } )
        else:
            #Not allowed to assign multiple procs the same local index
            # (presumably b/c there is no way to sub-divide the work 
            #  performed for a single index among multiple procs)
            if rank < nIndices:
                loc_indices = [ indices[rank] ]
            else:
                loc_indices = [ ] #extra procs do nothing
            owners = { indices[i]: i for i in range(nIndices) }
            
    else:
        nloc_std =  nIndices // nprocs
        extra = nIndices - nloc_std*nprocs # extra indices
          # so assign (nloc_std+1) indices to first extra procs
        if rank < extra:
            nloc = nloc_std+1
            nstart = rank * (nloc_std+1)
            loc_indices = [ indices[rank // (nloc_std+1)] ]
        else:
            nloc = nloc_std
            nstart = extra * (nloc_std+1) + (rank-extra)*nloc_std
        loc_indices = [ indices[i] for i in range(nstart,nstart+nloc)]

        owners = { } #which rank "owns" each index
        for r in range(extra):
            nstart = r * (nloc_std+1)
            for i in range(nstart,nstart+(nloc_std+1)):
                owners[indices[i]] = r
        for r in range(extra,nprocs):
            nstart = extra * (nloc_std+1) + (r-extra)*nloc_std
            for i in range(nstart,nstart+nloc_std):
                owners[indices[i]] = r

    return loc_indices, owners

def slice_up_range(n, num_slices, start=0):
    """ 
    Divides up `range(start,start+n)` into `num_slices` slices.

    Parameters
    ----------
    n : int
       The number of (consecutive) indices in the range to be divided.

    num_slices : int
       The number of slices to divide the range into.

    start : int, optional
       The starting entry of the range, so that the range to be 
       divided is `range(start,start+n)`.

    Returns
    -------
    list of slices
    """
    base = n // num_slices # base slice size
    m1 = n - base*num_slices # num base+1 size slices
    m2 = num_slices - m1     # num base size slices
    assert( ((base+1)*m1 + base*m2) == n )
    
    off = start
    ret =  [slice(off+(base+1)*i,off+(base+1)*(i+1)) for i in range(m1)]
    off += (base+1)*m1
    ret += [slice(off+base*i,off+base*(i+1)) for i in range(m2)]
    assert(len(ret) == num_slices)
    return ret



def distribute_slice(s, comm, allow_split_comm=True):
    """ 
    Partition a continuous slice evenly among `comm`'s processors.

    This function is similar to :func:`distribute_indices`, but
    is specific to the case when the indices being distributed
    are a consecutive set of integers (specified by a slice).

    Parameters
    ----------
    s : slice
        The slice to be partitioned.

    comm : mpi4py.MPI.Comm
        The communicator which specifies the number of processors and
        which may be split into returned sub-communicators.

    allow_split_comm : bool
        If True, when there are more processors than slice indices, 
        multiple processors will be given the *same* local slice
        and `comm` will be split into sub-communicators, one for each
        group of processors that are given the same local slice.
        If False, then "extra" processors are simply given
        nothing to do, i.e. an empty local slice.

    Returns
    -------
    loc_slice : slice
        A slice specifying the indices belonging to the current processor.
        
    loc_comm : mpi4py.MPI.Comm or None
        The local communicator for the group of processors which have been
        given the same `loc_slice` to compute, obtained by splitting `comm`.
        If `loc_slice` is unique to the current processor, or if 
        `allow_split_comm` is False, None is returned.
    """
    if comm is None:
        nprocs, rank = 1, 0
    else:
        nprocs = comm.Get_size()
        rank = comm.Get_rank()

    assert(s.step is None) #currently, no support for step != None slices
      # Though in principle this should be able to work
    if s.start is None or s.stop is None: indices = []
    else: indices = list(range(s.start,s.stop))
    loc_indices, _ = distribute_indices_base(indices, nprocs, rank,
                                                  allow_split_comm)
    if len(loc_indices) > 0:
        assert(loc_indices == list(range(loc_indices[0],loc_indices[-1]+1)))
        loc_slice = slice(loc_indices[0],loc_indices[-1]+1)
    
        #Split comm into sub-comms when there are more procs than
        # indices, resulting in all procs getting only a 
        # single index and multiple procs getting the *same*
        # (single) index.
        if nprocs > len(indices) and (comm is not None) and allow_split_comm:
            loc_comm = comm.Split(color=loc_indices[0], key=rank)  
        else: 
            loc_comm = None

    else: 
        loc_slice = slice(None)
        loc_comm = None


    return loc_slice, loc_comm


def gather_slices(slices, slice_owners,
                  arToFill, axis, comm, max_buffer_size=None):
    """ 
    Gathers data within a numpy array, `arToFill`, according to given slices.

    Upon entry it is assumed that the different processors within `comm` have
    computed different parts of `arToFill`, namely different slices of the
    `axis`-th axis.  At exit, data has been gathered such that all processors
    have the results for the entire `arToFill` (or at least for all the slices
    given).

    Parameters
    ----------
    slices : list
        A list of all the slices (computed by *any* of the processors, not
        just the current one).
        
    slice_owners : dict
        A dictionary mapping the index of a slice within `slices` to an 
        integer rank of the processor responsible for communicating that
        slice's data to the rest of the processors.

    arToFill : numpy.ndarray
        The array which contains partial data upon entry and the gathered
        data upon exit.

    axis : int
        The axis of `arToFill` on which the slices apply (which axis 
        do the slices in `slices` refer to?).

    comm : mpi4py.MPI.Comm or None
        The communicator specifying the processors involved and used
        to perform the gather operation.

    max_buffer_size : int or None
        The maximum buffer size in bytes that is allowed to be used 
        for gathering data.  If None, there is no limit.

    Returns
    -------
    None
    """
    if comm is None: return #no gathering needed!

    #Perform broadcasts for each slice in order
    my_rank = comm.Get_rank()
    arIndx = [ slice(None,None) ] * arToFill.ndim

    if max_buffer_size is not None: #no maximum of buffer size
        bytes_per_index = arToFill.nbytes / arToFill.shape[axis]
        max_indices = max(1,int(max_buffer_size/bytes_per_index))
        #bShowMessage = bool(my_rank == 0)
    else:
        max_indices = None
        #bShowMessage = False

    for iSlice,slc in enumerate(slices):
        owner = slice_owners[iSlice] #owner's rank
        assert(slc.step is None or slc.step == 1) #only allow step=1 slices (easy stop-start arithmetic below)

        if max_indices is None or max_indices >= (slc.stop-slc.start):
            arIndx[axis] = slc
            buf = arToFill[arIndx].copy() if (my_rank == owner) \
                else _np.empty(arToFill[arIndx].shape, arToFill.dtype)
            comm.Bcast(buf, root=slice_owners[iSlice])
            if my_rank != owner: arToFill[arIndx] = buf
        else:
#            if bShowMessage:
#                print("MPIDB: gather_slices restricting %s to %d indices at once"
#                      % (str(slc),max_indices)); bShowMessage = False
            sub_start = slc.start
            while sub_start < slc.stop: #broadcast in chunks to keep buffer size small
                sub_slc = slice(sub_start, min(sub_start+max_indices,slc.stop))
                arIndx[axis] = sub_slc
                buf = arToFill[arIndx].copy() if (my_rank == owner) \
                    else _np.empty(arToFill[arIndx].shape, arToFill.dtype)
                comm.Bcast(buf, root=slice_owners[iSlice])
                if my_rank != owner: arToFill[arIndx] = buf
                sub_start += max_indices


def distribute_for_dot(contracted_dim, comm):
    """
    Prepares for one or muliple distributed dot products given the dimension
    to be contracted (i.e. the number of columns of A or rows of B in dot(A,B)).
    The returned slice should be passed as `loc_slice` to :func:`mpidot`.

    Parameters
    ----------
    contracted_dim : int
        The dimension that will be contracted in ensuing :func:`mpidot`
        calls (see above).

    comm : mpi4py.MPI.Comm or None
        The communicator used to perform the distribution.

    Returns
    -------
    slice
        The "local" slice specifying the indices belonging to the current
        processor.  Should be passed to :func:`mpidot` as `loc_slice`.
    """
    loc_indices,_,_ = distribute_indices(
        list(range(contracted_dim)), comm, False)

    #Make sure local columns are contiguous
    start,stop = loc_indices[0], loc_indices[-1]+1
    assert(loc_indices == list(range(start,stop)))
    return slice(start, stop) # local column range as a slice

def mpidot(a,b,loc_slice,comm):
    """
    Performs a distributed dot product, dot(a,b).

    Parameters
    ----------
    a,b : numpy.ndarray
        Arrays to dot together.

    loc_slice : slice
        A slice specifying the indices along the contracted dimension belonging
        to this processor (obtained from :func:`distribute_for_dot`)

    comm : mpi4py.MPI.Comm or None
        The communicator used to parallelize the dot product.

    Returns
    -------
    numpy.ndarray
    """
    if comm is None or comm.Get_size() == 1:
        assert(loc_slice == slice(0,b.shape[0]))
        return _np.dot(a,b)

    from mpi4py import MPI #not at top so can import pygsti on cluster login nodes
    loc_dot = _np.dot(a[:,loc_slice],b[loc_slice,:])
    result = _np.empty( loc_dot.shape, loc_dot.dtype )
    comm.Allreduce(loc_dot, result, op=MPI.SUM)

    #DEBUG: assert(_np.linalg.norm( _np.dot(a,b) - result ) < 1e-6)
    return result

    #myNCols = loc_col_slice.stop - loc_col_slice.start
    ## Gather pieces of coulomb tensor together
    #nCols = comm.allgather(myNCols)  #gather column counts into an array
    #displacements = _np.concatenate(([0],_np.cumsum(sizes))) #calc displacements
    #
    #result = np.empty(displacements[-1], a.dtype)
    #comm.Allgatherv([CTelsLoc, size, MPI.F_DOUBLE_COMPLEX], \
    #                [CTels, (sizes,displacements[:-1]), MPI.F_DOUBLE_COMPLEX])

    