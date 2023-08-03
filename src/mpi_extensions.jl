gather(obj, comm::MPI.Comm, root::Integer=Cint(0)) = gather(obj, root, comm)
function gather(obj, root::Integer, comm::MPI.Comm)
  isroot = MPI.Comm_rank(comm) == root
  count = Ref{Clong}()
  buf = MPI.serialize(obj)
  count[] = length(buf)
  counts = MPI.Gather(count[], root, comm)
  if isroot
    rbuf = Array{UInt8}(undef, reduce(+, counts))
    rbuf = MPI.VBuffer(rbuf, counts)
  else
    rbuf = nothing
  end
  MPI.Gatherv!(buf, rbuf, root, comm)
  if isroot
    objs = []
    for v in 1:length(rbuf.counts)
      endind = v == length(rbuf.counts) ? length(rbuf.data) : rbuf.displs[v + 1] + 1
      startind = rbuf.displs[v] + 1
      push!(objs, MPI.deserialize(rbuf.data[startind:endind]))
    end
  else
    objs = nothing
  end
  return objs
end

function psi_bcast(psi0::MPS, splits, root::Integer, comm)
  @assert isinteger(length(psi0)/splits)
  step = Int(length(psi0)/splits)

  isroot = MPI.Comm_rank(comm) == root
  bc_data = Vector{Vector{ITensor}}(undef, splits)
  ortholims = MPI.bcast(ortho_lims(psi0), root, comm)

  for i in 1:splits
      bc_data[i] = MPI.bcast(psi0[(i-1)*step+1:i*step], root, comm)
  end
  if !isroot
      psi0 = nothing
      psi0 = MPS([(bc_data...)...]; ortho_lims = ortholims)
  end
  return psi0
end

function psi_bcast(psi0::MPS, root::Integer, comm)
  return psi_bcast(psi0, length(psi0), root, comm)
end

function bcast(obj, root::Integer, comm::MPI.Comm)
  isroot = MPI.Comm_rank(comm) == root
  count = Ref{Clong}()
  if isroot
    buf = MPI.serialize(obj)
    count[] = length(buf)
  end
  MPI.Bcast!(count, root, comm)
  if !isroot
    buf = Array{UInt8}(undef, count[])
  end
  MPI.Bcast!(buf, root, comm)
  if !isroot
    obj = MPI.deserialize(buf)
  end
  return obj
end

function allreduce(sendbuf, op, comm::MPI.Comm)
  ##maybe better to implement as allgather with local reduce, but higher communication cost associated
  bufs = gather(sendbuf, 0, comm)
  rank = MPI.Comm_rank(comm)
  if rank == 0
    res = reduce(op, bufs)
  else
    res = nothing
  end
  return bcast(res, 0, comm)
end
