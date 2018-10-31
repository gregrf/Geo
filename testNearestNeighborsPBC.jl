using Test
using Random: seed!

using NearestNeighborsPBC

function neighbor_sets_equal(S, T)
    return all(Set(s) == Set(t) for (s, t) in zip(S, T))
end

@testset "test NearestNeighborsPBC against naive N^2 implementation. with r_max passed" begin
    seed!(2354523)
    L = 10.0
    res = []
    for _i in 1:20
        for r_max in [0.1, 0.3, 1.0, 1.5, 9.]
            for ndim in [2, 3]
                for n  in [4, 10]
                    X = L .* rand(Float64, (ndim, n,))

                    ind1 = inrangePBC_naive(X, L, r_max)

                    kdt = KDTreePBC(X, L, r_max=r_max)
                    ind2 = inrange(kdt, X, r_max)
                    
                    push!(res, neighbor_sets_equal(ind1, ind2))
                end
            end
        end
    end
    @test all(res)
end

@testset "test NearestNeighborsPBC against naive N^2 implementation. without passing r_max and using Float32's." begin
    seed!(2354524)
    L = 6.4f0
    res = []
    for _i in 1:20
        for r_max in [0.1f0, 0.3f0, 1.0f0, 1.5f0, 4.0f0]
            for ndim in [2, 3]
                for n  in [4, 10]
                    X = L .* rand(Float32, (ndim, n,))

                    ind1 = inrangePBC_naive(X, L, r_max)

                    kdt = KDTreePBC(X, L)
                    ind2 = inrange(kdt, X, r_max)
                    
                    push!(res, neighbor_sets_equal(ind1, ind2))
                end
            end
        end
    end
    @test all(res)
end

