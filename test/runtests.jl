using Test

import TroyonBetaNN as TBNN
using IMAS

@testset "TroyonBetaNN.jl" begin

    file_path = joinpath(@__DIR__, "data", "dd_D3D.json")
    dd = IMAS.json2imas(file_path);

    # run & show simple terminal output
    TD_vec = TBNN.Calculate_Troyon_beta_limits_for_IMAS_dd(dd);

    # run & show verbose terminal output
    TD_vec = TBNN.Calculate_Troyon_beta_limits_for_IMAS_dd(dd; verbose=true);

    TD=TD_vec[2]
    @testset "Check MLP's βₙ_limits with references" begin
        @test TD.MLPs[1].βₙ_limit ≈ 3.5083038793808123 atol=1e-6
        @test TD.MLPs[2].βₙ_limit ≈ 3.521892957155812 atol=1e-6
        @test TD.MLPs[3].βₙ_limit ≈ 3.9271056622431315 atol=1e-6
    end

end
