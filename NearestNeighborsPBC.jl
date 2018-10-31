"""
NearestNeighborsPBC is a module for using NearestNeighbors kdtree implementation to find neighbors of points in a periodic square box. This is done by simply mirroring particles across boundaries, using the NearestNeighbors's kdtree on that larger set of points, then relabeling indices to correspond to the original particles. The only method currently implemented on the new periodic kdtrees is inrange(), which can be used analogously to the inrange() defined in NearestNeighbors. From a user perspective the only differences are than KDTreePBC must be passed a box size, `L`, and may optionally be passed a maximum distance, `r_max`, over which neighbors will be sought. Also, `X` must be passed as a ndim x N matrix. Setting `r_max` will not change the results but will avoid mirroring distant points if `r_max < L` increasing the speed of the calculation.

Currently NearestNeighborsPBC.KDTreePBC is only coded to use Euclidean norms, however, it can accept sets of points in any dimension and `X`, `L`, and `r_max` arguments of any AbstractFloat type, however, all arguments must be of the same type.
"""
module NearestNeighborsPBC
using LinearAlgebra: norm_sqr
using NearestNeighbors
using StaticArrays

export KDTreePBC, inrange, inrangePBC_naive, smallest_separation_PBC

"""
    smallest_separation_PBC(X1, X2, L::Real)

Return the result of `X2 .- X1` subject to periodic boundary conditions in a box from `0` to `L`
along every axis. Since the elements of these points are only defined modulo `L`, the result
of the subtraction is multivalued. This function returns the separation with the smallest norm.
The elements of the result will all be bounded between `-L/2` and `L/2`.
"""
function smallest_separation_PBC(X1, X2, L::Real)
    return @. mod(X1 - X2 + L / 2, L) - L / 2
end

"""
    mirror_axes!(list::AbstractVector, mirror_pids::Vector{Int}, pid::Int, point::AbstractVector, mirror_along::Vector{Int}, L::Real, pad_distance::Real)

mirror `point` across `0` or `L` according to whether elements in `mirror_along` are -1 (mirror element across 0),
0 (don't mirror element), 1 (mirror element across `L`), pushing mirrored point to list and the pid of the point
it mirrors to `mirror_pids`. `L` sets the size of the container and d the distance from the boundary to be mirrored. Note that points in the interior are trivally mirrored when `mirror_along` is [0, 0, ...] an pushed once to list which is a flattened list of points.
"""
function mirror_axes!(list::AbstractVector, mirror_pids::Vector{Int}, pid::Int, point::AbstractVector, mirror_along::Vector{Int}, L::Real, pad_distance::Real)
    ndim = length(point)
    d = pad_distance
    mirror_point = copy(point)
    write = falses(ndim)

    for (ax, low_no_highQ) in enumerate(mirror_along)
        el = point[ax]
        if low_no_highQ == -1 && el < d
            mirror_point[ax] += L
            write[ax] = true
        elseif low_no_highQ == 0
            write[ax] = true
        elseif low_no_highQ == 1 && el > L - d
            mirror_point[ax] -= L
            write[ax] = true
        end
    end
    if all(write)
        push!(mirror_pids, pid)
        append!(list, mirror_point)
    end
end

"""
    function mirror_points_periodically(X::AbstractMatrix, L::Real, r_max::Real)

Apply periodic bounary conditions to points, `X=[[x0, y0,...] [x1, y1, ...] ...]`, which are forced by a modulo operation to live in a periodic square with `0<=xi<L`, by mirroring points within `r_max` of a boundary along an of the axes. The returned points will then have coordinates `-r_max < xi < L + r_max`.

Returns `mirror_pids`, `points_incl_mirrors` where `mirror_pids` are indices of the mirrored points in the original list of points and `points_incl_mirrors` is the set of points expanded to include mirrored points.
"""
function mirror_points_periodically(X::AbstractMatrix, L::Real, r_max::Real)
    ndim, n = size(X)
    
    X0 = mod.(X, L)
    mirror_points_list = collect(reshape(X0, :))
    mirror_pids = Vector{Int}(1:n)

    outer_prod = filter((x)->norm_sqr(x) > 0.5, map(collect, Iterators.product([[-1, 0, 1] for i in 1:ndim]...)))
    
    for pid in 1:n
        for mirror_along in outer_prod
            mirror_axes!(mirror_points_list, mirror_pids,
                         pid, X0[:, pid], mirror_along, L, r_max)
        end
    end

    mirror_points_list = reshape(mirror_points_list, (ndim, :))
    return mirror_pids, mirror_points_list
end

"""
    KDTreePBC(X::AbstractMatrix, L::Real; r_max::Real=-1.0)

Create a KD tree for points, `X=[[x0, y0,...] [x1, y1, ...] ...]`, subject to periodic bounary conditions within a periodic square with `0<=xi<L`. This is accomplished by mirroring points within `r_max` of a boundary along an of the axes. `r_max` should be specified if neighbors will be sought within distance significantly less than `L` as this minimizes the number of points that need to be mirrored.


To return a vector of vectors of neighbors of `X`, use `inrange(kdt::KDTreePBC, X, r_max)`.
"""
struct KDTreePBC{T, ndim}
    kdtree::KDTree{SArray{Tuple{ndim},T,1,ndim},Euclidean, T}
    indices_of_mirror_points::Vector{Int}
    r_max::T
    function KDTreePBC(X::Matrix{T}, L::T; r_max::T=T(-1)) where {T<:AbstractFloat}
        if r_max < 0
            r_max = L
        end
        if r_max > L
            throw(ErrorException("r_max must not be larger than L"))
        elseif L <=0
            throw(ErrorException("L must be > 0"))
        end
        mirror_pids, X_mirror = mirror_points_periodically(X, L, r_max)
        kdtree = KDTree(X_mirror)
        new{T, size(X, 1)}(kdtree, mirror_pids, r_max)
    end
end

"""
    inrange(kdt::KDTreePBC{T, ndim}, X::Matrix{T}, r_max::T) where {T, ndim}

Return a vector of vectors of neighbors of `X` within distance, `r_max`, anologously to NearestNeighbors' inrange() method only subject to periodic boundary conditions within a box of size `L` which is passed to KDTreePCB(). Passing `r_max` when building the KDTreePBC will not effect the result but will speed up the calculation when `r_max << L`.

For the naive N^2 version of the calculation, see
inrangePBC_naive(X::AbstractMatrix, L::Real, r_max::Real).
"""
function inrange(kdt::KDTreePBC{T, ndim}, X::Matrix{T}, r_max::T) where {T, ndim}
    mirror_pids = kdt.indices_of_mirror_points
    idxs = NearestNeighbors.inrange(kdt.kdtree, X, r_max)

    if r_max > kdt.r_max
        throw(ErrorException("r_max cannot be larger than kdt.r_max."))
    end

    contact_list = Vector{Int}[]
    for (pid, neigh_list) in enumerate(idxs)
        _list = Int[]
        for mi in neigh_list
            i = mirror_pids[mi]
            if i != pid
                push!(_list, i)
            end
        end
        push!(contact_list, _list)
    end
    return contact_list
end

"""
    inrangePBC_naive(X::AbstractMatrix, L::Real, r_max::Real)

calculate the neighbors subject to periodic boundary conditions
equivalent to `inrange(KDTreePBC(X, L, r_max=r_max), X, L, r_max)`
by the naive `N^2` algorithm of the distance between all pairs of points
(where `N=size(X, 2)`).

This algorithm should be faster for very small `N`, however,
it is primarily intended to be used to test `KDTreePBC` and
`inrange(kdt::KDTreePBC, L, rmax)`.
"""
function inrangePBC_naive(X::AbstractMatrix, L::Real, r_max::Real)
    r_max_sq = r_max^2
    
    neighbor_list = Vector{Int}[]
    for i in 1:size(X, 2)
        _list = Int[]
        for j in 1:size(X, 2)
            if i != j && norm_sqr(
                    smallest_separation_PBC(X[:, i], X[:, j], L)
                    ) < r_max_sq
                push!(_list, j)
            end
        end
        push!(neighbor_list, _list)
    end
    return neighbor_list
end

end#module
