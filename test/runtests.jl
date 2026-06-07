using Test

import TroyonBetaNN as TBNN
using IMAS

include("references.jl")

@testset "TroyonBetaNN.jl" begin

    file_path = joinpath(pkgdir(TBNN), "test", "data", "dd_D3D.json")
    dd = IMAS.json2imas(file_path);

    # run & show simple terminal output
    TD_vec = TBNN.calculate_Troyon_beta_limits_for_IMAS_dd(dd);

    # run & show verbose terminal output.
    # verbose=true savefig's sample-point plots to the CWD, so run with the CWD
    # set to a temp dir to keep the repo clean (the plots aren't asserted).
    TD_vec = cd(mktempdir()) do
        TBNN.calculate_Troyon_beta_limits_for_IMAS_dd(dd; verbose=true)
    end

    TD = TD_vec[2]
    @testset "Check MLP's βₙ_limits with references" begin
        @test isapprox(TD.MLPs[1].βₙ_limit, 3.5083038793808123; atol=1e-6)
        @test isapprox(TD.MLPs[2].βₙ_limit, 3.521892957155812; atol=1e-6)
        @test isapprox(TD.MLPs[3].βₙ_limit, 3.9271056622431315; atol=1e-6)
    end

    @testset "Interpolation outputs (sample points)" begin
        @test isapprox(TD.sampPoints.q,        Q_REF;        rtol=1e-12)
        @test isapprox(TD.sampPoints.pressure, PRESSURE_REF; rtol=1e-12)
        @test isapprox(TD.sampPoints.R,        R_REF;        rtol=1e-12)
        @test isapprox(TD.sampPoints.Z,        Z_REF;        rtol=1e-12)
    end

end
