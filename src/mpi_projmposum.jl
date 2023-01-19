function MPI.RBuffer(senddata::ITensor, recvdata::ITensor)
  senddata = permute(senddata, inds(recvdata); allow_alias=true)
  @assert inds(senddata) == inds(recvdata)
  _senddata = ITensors.data(senddata)
  _recvdata = ITensors.data(recvdata)
  @assert length(_senddata) == length(_recvdata)
  return MPI.RBuffer(_senddata, _recvdata)
end

struct MPISum{T}
  data::T
  comm::MPI.Comm
end

MPISum(data::T, comm=MPI.COMM_WORLD) where {T} = MPISum{T}(data, comm)

nsite(P::MPISum) = ITensors.nsite(P.data)

Base.length(P::MPISum) = length(P.data)

function _Allreduce(sendbuf::ITensor, op, comm)
  return _Allreduce(NDTensors.storagetype(tensor(sendbuf)), sendbuf, op, comm)
end

function _Allreduce(::Type{<:BlockSparse}, sendbuf::ITensor, op, comm)  @deprecate
  # Similar to:
  #
  # recvbuf = ITensor(eltype(sendbuf), flux(sendbuf), inds(sendbuf))
  #
  # But more efficient.
  nprocs = MPI.Comm_size(comm)
  N = order(sendbuf)
  @show MPI.Comm_rank(comm), nnzblocks(sendbuf)
  max_nblocks = maximum(MPI.Allgather(nnzblocks(sendbuf), comm))
  blocks_sendbuf = fill(0, max_nblocks, N)
  nzblocks_matrix = Matrix{Int}(reduce(hcat, collect.(nzblocks(sendbuf)))')
  blocks_sendbuf[1:size(nzblocks_matrix, 1), :] = nzblocks_matrix
  allblocks = MPI.Allgather(blocks_sendbuf, comm)
  allblocks_tensor = reshape(allblocks, max_nblocks, N, nprocs)
  allblocks_vector = [allblocks_tensor[:, :, i] for i in 1:nprocs]
  allblocks_vector_tuple = [
    [Tuple(allblocks_vector[n][i, :]) for i in 1:max_nblocks] for n in 1:nprocs
  ]
  blocks = Block.(filter(≠(ntuple(Returns(0), N)), union(allblocks_vector_tuple...)))
  recvbuf = itensor(BlockSparseTensor(eltype(sendbuf), blocks, inds(sendbuf)))
  sendbuf_filled = copy(recvbuf)
  sendbuf_filled .= sendbuf
  MPI.Allreduce!(sendbuf_filled, recvbuf, +, comm)
  return recvbuf
end

function _Allreduce(::Type{<:Dense}, sendbuf::ITensor, op, comm)
  recvbuf = similar(sendbuf)
  MPI.Allreduce!(sendbuf, recvbuf, op, comm)
  return recvbuf
end

function _allreduce(sendbuf,op,comm::MPI.Comm)
  ##maybe better to implement as allgather with local reduce, but higher communication cost associated
  bufs=MPI.gather(sendbuf,0,comm)
  rank=MPI.Comm_rank(comm)
  if rank==0
    if op==+
      res=sum(bufs)
    else
      res=reduce(op,bufs)
    end
  else
    res=nothing
  end
  return MPI.bcast(res,0,comm)
end

function product(P::MPISum, v::ITensor)
  return _allreduce(P.data(v),+,P.comm)
end

function Base.eltype(P::MPISum)
  return eltype(P.data)
end

(P::MPISum)(v::ITensor) = product(P, v)

Base.size(P::MPISum) = size(P.data)


function position!(P::MPISum{T}, psi::MPS, pos::Int)  where T<:ITensors.AbstractProjMPO
  makeL!(P, psi, pos - 1)
  makeR!(P, psi, pos + nsite(P))
  return P
end

function makeL!(P::MPISum, psi::MPS, k::Int)
  _makeL!(P, psi, k)
  return P
end

function makeR!(P::MPISum, psi::MPS, k::Int)
  _makeR!(P, psi, k)
  return P
end



function _makeL!(_P::MPISum, psi::MPS, k::Int)::Union{ITensor,Nothing}
  # Save the last `L` that is made to help with caching
  # for DiskProjMPO
  P=_P.data
  ll = P.lpos
  if ll ≥ k
    # Special case when nothing has to be done.
    # Still need to change the position if lproj is
    # being moved backward.
    P.lpos = k
    return nothing
  end
  # Make sure ll is at least 0 for the generic logic below
  ll = max(ll, 0)
  L = lproj(P)
  while ll < k
    # sync the linkindex across processes
    mylink=linkind(psi,ll+1)
    otherlink=MPI.bcast(mylink,0,_P.comm)
    replaceind!(psi[ll+1],mylink,otherlink)
    replaceind!(psi[ll+2],mylink,otherlink)
    
    L = L * psi[ll + 1] * P.H[ll + 1] * dag(prime(psi[ll + 1]))
    P.LR[ll + 1] = L
    ll += 1
  end
  # Needed when moving lproj backward.
  P.lpos = k
  return L
end

function _makeR!(_P::MPISum{ProjMPO}, psi::MPS, k::Int)::Union{ITensor,Nothing}
  # Save the last `R` that is made to help with caching
  # for DiskProjMPO
  P=_P.data
  rl = P.rpos
  if rl ≤ k
    # Special case when nothing has to be done.
    # Still need to change the position if rproj is
    # being moved backward.
    P.rpos = k
    return nothing
  end
  N = length(P.H)
  # Make sure rl is no bigger than `N + 1` for the generic logic below
  rl = min(rl, N + 1)
  R = rproj(P)
  while rl > k
    #sync linkindex across processes
    mylink=linkind(psi,rl-2)
    otherlink=MPI.bcast(mylink,0,_P.comm)
    replaceind!(psi[rl-2],mylink,otherlink)
    replaceind!(psi[rl-1],mylink,otherlink)
    
    R = R * psi[rl - 1] * P.H[rl - 1] * dag(prime(psi[rl - 1]))
    P.LR[rl - 1] = R
    rl -= 1
  end
  P.rpos = k
  return R
end



function noiseterm(P::MPISum, phi::ITensor, dir::String)
  ##ToDo: I think the logic here is wrong.
  ##The noiseterm should be calculated on the reduced P*v and then broadcasted?
  return _allreduce(noiseterm(P.data, phi, dir), +, P.comm)
end
