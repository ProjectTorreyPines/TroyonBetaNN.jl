module TroyonBetaNN


using IMAS
using JSON
using BSplineKit
using Printf
using Plots
using PrettyTables

import ONNXRunTime as ORT

export Troyon_Data, Load_predefined_Troyon_NN_Models

@kwdef mutable struct MLP_Model
    n::Int # toroidal mode number
    W::Matrix{Float64} # map from input to hidden layer
    V::Vector{Float64} # map from hidden layer to output
    βₙ_limit::Float64=NaN # scalar output
    input_file::String=""
end

@kwdef mutable struct CNN_Model
    n::Int=1 # toroidal mode number
    model::ORT.InferenceSession
    input::Dict=Dict()
    βₙ_limit::Float64=NaN # scalar output
    input_file::String=""
end

@kwdef mutable struct Sample_Points
    R::Vector{Float64}=[]
    Z::Vector{Float64}=[]
    ψₙ::Vector{Float64}=[0:0.1:0.9; 0.95; 0.975]
    q::Vector{Float64}=[]
    pressure::Vector{Float64}=[]
end

@kwdef mutable struct Troyon_Data
    sampPoints :: Sample_Points=[] # Sample Points for NN
    MLPs :: Vector{MLP_Model}=[] # NN (MLP) models for (n=1,2,3) modes
    CNN :: CNN_Model=[] # CNN models for n=1 mode
end


function Load_predefined_Troyon_NN_Models(MLP_file::String =joinpath(@__DIR__, "../data/MLP_Model.json"), CNN_file::String =joinpath(@__DIR__, "../data/CNN_Model.onnx"))
    # Read MLP file
    data_from_file = JSON.parsefile(MLP_file)

    target_n_modes = [1,2,3]

    MLPs = Vector{MLP_Model}(undef, 3)
    for n in target_n_modes
        w_data = data_from_file[n]["W"]
        w_data = Float64.(hcat(w_data...)')

        v_data = Float64.(data_from_file[n]["V"])

        # Create MLP instance
        MLPs[n] = MLP_Model(n, w_data, v_data, NaN, MLP_file)
    end

    # Read CNN file
    CNN = CNN_Model(model= ORT.load_inference(CNN_file));

    return Troyon_Data(Sample_Points(), MLPs, CNN)
end


function Load_predefined_Troyon_MLP_Model(file_path :: String =joinpath(@__DIR__, "../data/MLP_Model.json"))
    data_from_file = JSON.parsefile(file_path)

    target_n_modes = [1,2,3]

    MLPs = Vector{MLP_Model}(undef, 3)
    for n in target_n_modes
        w_data = data_from_file[n]["W"]
        w_data = Float64.(hcat(w_data...)')

        v_data = Float64.(data_from_file[n]["V"])

        # Create MLP instance
        MLPs[n] = MLP_Model(n, w_data, v_data, NaN, file_path)
    end

    return Troyon_Data(Sample_Points(), MLPs)
end

function Calculate_Troyon_beta_limits_for_IMAS_dd(dd::IMAS.dd; kwargs...)
    Neqt = length(dd.equilibrium.time_slice)
    TD_vec = [Load_predefined_Troyon_NN_Models() for _ in 1:Neqt]

    Calculate_Troyon_beta_limits_for_IMAS_dd(TD_vec, dd; kwargs...)
    return TD_vec
end

function Calculate_Troyon_beta_limits_for_IMAS_dd(TD_vec::Vector{Troyon_Data}, dd::IMAS.dd; kwargs...)
    verbose = get(kwargs, :verbose, false)

    yellow_bold = Crayon(foreground=:yellow, bold=true)
    for tid = 1:length(dd.equilibrium.time_slice)
        this_eqt = dd.equilibrium.time_slice[tid]

        if isnan(this_eqt.global_quantities.vacuum_toroidal_field.b0)
            @warn(@sprintf("Equilibrium time_slice #%d has no equilirbium information\nSkipping Troyon βₙ calculations ...\n", tid));
        else
            println(yellow_bold(@sprintf("\nFor equilibrium time_slice #%d @ t=%.2g secs", tid, this_eqt.time)))
            Calculate_Troyon_beta_limits_for_a_given_time_slice(TD_vec[tid], this_eqt; kwargs...)
        end
    end
end

function Calculate_Troyon_beta_limits_for_a_given_time_slice(eqt::IMAS.equilibrium__time_slice; kwargs...)
    TD = Load_predefined_Troyon_NN_Models()
    Calculate_Troyon_beta_limits_for_a_given_time_slice(TD, eqt; kwargs...)
    return TD
end

function Calculate_Troyon_beta_limits_for_a_given_time_slice(TD::Troyon_Data, eqt::IMAS.equilibrium__time_slice; kwargs...)
    if isnan(eqt.global_quantities.vacuum_toroidal_field.b0)
        @warn("Given time_slice has no equilirbium information\nSkipping Troyon βₙ calculations ...\n");
        return
    end


    Check_validity_of_NN_for_given_input(TD, eqt; kwargs...)

    Sample_points_from_equilibrium(TD, eqt)

    # First, MLP model
    # Calculate 42 neurons from sample Points on equilibrium
    X_neurons= _calculate_MLP_neurons(TD, eqt)

    # Calculate Troyon beta_N limits using MLP model
    for MLP in TD.MLPs
        X = [X_neurons; 1] # add a bias neuron (total 43 neurons)

        # activate hidden neurons
        Y = 1.0 ./ (1.0 .+ exp.(-MLP.W * X))
        Y = [Y; 1] # add a bias neuron

        MLP.βₙ_limit =  (MLP.V') * Y
    end

    # Calculate Troyon beta_N limits (n=1) using CNN model
    _set_CNN_input_neurons_from_sampled_points(TD, eqt)
    CNN_output = TD.CNN.model(TD.CNN.input)["tf.math.multiply"];
    TD.CNN.βₙ_limit = Float64.(vec(CNN_output)[1]);


    equilibrium_βₙ = eqt.global_quantities.beta_normal
    # _print_results_to_stdout(TD; eq_betaN=equilibrium_βₙ ,kwargs...)

    _print_results_to_stdout(TD.MLPs; eq_betaN=equilibrium_βₙ ,kwargs...)
    _print_results_to_stdout(TD.CNN; eq_betaN=equilibrium_βₙ ,kwargs...)

    verbose = get(kwargs, :verbose, false)
    if verbose
        plot_sample_points(TD, eqt; file_type="png")
    end
end

function _print_results_to_stdout(TD; kwargs...)
    verbose = get(kwargs, :verbose, false)
    eq_betaN = get(kwargs, :eq_betaN, -1.0)

    if verbose && eq_betaN > 0
        white_bold = Crayon(foreground=:white, bold=true)
        blue_bold = Crayon(foreground=:blue, bold=true)

        header = ["Tor. mode", "Troyon βₙ Limit", @sprintf("Equilibrium (βₙ=%.2f)",eq_betaN)]

        MLP_stability_vec = Vector{String}(undef, length(TD.MLPs),)
        for (n, MLP) in pairs(TD.MLPs)
            if (eq_betaN > MLP.βₙ_limit)
                MLP_stability_vec[n] = "Unstable"
            elseif eq_betaN > 0.95*MLP.βₙ_limit
                MLP_stability_vec[n] = "Marginal"
            else
                MLP_stability_vec[n] = "Stable"
            end
        end

        data = hcat(getfield.(TD.MLPs,:n), getfield.(TD.MLPs,:βₙ_limit), MLP_stability_vec)

        hl1 = Highlighter( (data, i, j) -> (j in (1,3)) && (data[i, end]=="Unstable"), crayon"red bold");
        hl2 = Highlighter( (data, i, j) -> (j in (1,3)) && (data[i, end]=="Marginal"), crayon"yellow bold");
        hl3 = Highlighter( (data, i, j) -> (j==3) && (data[i, end]=="Stable"), crayon"green");

        println("\n",blue_bold("MLP: "), white_bold("(Troyon βₙ Limits) vs (Equilibrium "), blue_bold(@sprintf("βₙ=%.2f",eq_betaN)), ")")
        pretty_table(
            data;
            formatters    = ft_printf("%5.3f", 2:4),
            header        = header,
            header_crayon = crayon"white bold",
            highlighters  = (hl1, hl2, hl3),
            tf            = tf_unicode_rounded
        )

        if (eq_betaN > TD.CNN.βₙ_limit)
            CNN_stabiltiy = "Unstable"
        elseif eq_betaN > 0.95*TD.CNN.βₙ_limit
            CNN_stabiltiy = "Marginal"
        else
            CNN_stabiltiy = "Stable"
        end
        data = hcat(TD.CNN.n, TD.CNN.βₙ_limit, CNN_stabiltiy)

        hl1 = Highlighter( (data, i, j) -> (j in (1,3)) && (data[i, end]=="Unstable"), crayon"red bold");
        hl2 = Highlighter( (data, i, j) -> (j in (1,3)) && (data[i, end]=="Marginal"), crayon"yellow bold");
        hl3 = Highlighter( (data, i, j) -> (j==3) && (data[i, end]=="Stable"), crayon"green");

        println("\n",blue_bold("CNN: "), white_bold("(Troyon βₙ Limits) vs (Equilibrium "), blue_bold(@sprintf("βₙ=%.2f",eq_betaN)), ")")
        pretty_table(
            data;
            formatters    = ft_printf("%5.3f", 2:4),
            header        = header,
            header_crayon = crayon"white bold",
            highlighters  = (hl1, hl2, hl3),
            tf            = tf_unicode_rounded
        )

    else
        @printf("\nTroyon (no-wall) Beta Limits\n")
        @printf("(MLP model):\n")
        for this_MLP in TD.MLPs
            @printf("↳ (n=%d): βₙ=%.3f\n", this_MLP.n, this_MLP.βₙ_limit)
        end

        @printf("\n(CNN model):\n")
        @printf("↳ (n=%d): βₙ=%.3f\n", TD.CNN.n, TD.CNN.βₙ_limit)
    end
end


function _print_results_to_stdout(MLPs::Vector{MLP_Model}; kwargs...)
    verbose = get(kwargs, :verbose, false)
    eq_betaN = get(kwargs, :eq_betaN, -1.0)

    if verbose && eq_betaN > 0
        white_bold = Crayon(foreground=:white, bold=true)
        blue_bold = Crayon(foreground=:blue, bold=true)

        header = ["Tor. mode", "Troyon βₙ Limit", @sprintf("Equilibrium (βₙ=%.2f)",eq_betaN)]

        MLP_stability_vec = Vector{String}(undef, length(MLPs),)
        for (n, MLP) in pairs(MLPs)
            if (eq_betaN > MLP.βₙ_limit)
                MLP_stability_vec[n] = "Unstable"
            elseif eq_betaN > 0.95*MLP.βₙ_limit
                MLP_stability_vec[n] = "Marginal"
            else
                MLP_stability_vec[n] = "Stable"
            end
        end

        data = hcat(getfield.(MLPs,:n), getfield.(MLPs,:βₙ_limit), MLP_stability_vec)

        hl1 = Highlighter( (data, i, j) -> (j in (1,3)) && (data[i, end]=="Unstable"), crayon"red bold");
        hl2 = Highlighter( (data, i, j) -> (j in (1,3)) && (data[i, end]=="Marginal"), crayon"yellow bold");
        hl3 = Highlighter( (data, i, j) -> (j==3) && (data[i, end]=="Stable"), crayon"green");

        println("\n",blue_bold("MLP: "), white_bold("(Troyon βₙ Limits) vs (Equilibrium "), blue_bold(@sprintf("βₙ=%.2f",eq_betaN)), ")")
        pretty_table(
            data;
            formatters    = ft_printf("%5.3f", 2:4),
            header        = header,
            header_crayon = crayon"white bold",
            highlighters  = (hl1, hl2, hl3),
            tf            = tf_unicode_rounded
        )
    else
        @printf("\n(MLP model): Troyon Beta_N Limits\n")
        for this_MLP in MLPs
            @printf("↳ (n=%d): βₙ=%.3f\n", this_MLP.n, this_MLP.βₙ_limit)
        end
    end
end

function _print_results_to_stdout(CNN::CNN_Model; kwargs...)
    verbose = get(kwargs, :verbose, false)
    eq_betaN = get(kwargs, :eq_betaN, -1.0)

    if verbose && eq_betaN > 0
        white_bold = Crayon(foreground=:white, bold=true)
        blue_bold = Crayon(foreground=:blue, bold=true)

        header = ["Tor. mode", "Troyon βₙ Limit", @sprintf("Equilibrium (βₙ=%.2f)",eq_betaN)]

        if (eq_betaN > CNN.βₙ_limit)
            stability = "Unstable"
        elseif eq_betaN > 0.95*CNN.βₙ_limit
            stability = "Marginal"
        else
            stability = "Stable"
        end
        data = hcat(CNN.n, CNN.βₙ_limit, stability)

        hl1 = Highlighter( (data, i, j) -> (j in (1,3)) && (data[i, end]=="Unstable"), crayon"red bold");
        hl2 = Highlighter( (data, i, j) -> (j in (1,3)) && (data[i, end]=="Marginal"), crayon"yellow bold");
        hl3 = Highlighter( (data, i, j) -> (j==3) && (data[i, end]=="Stable"), crayon"green");

        println("\n",blue_bold("CNN: "), white_bold("(Troyon βₙ Limits) vs (Equilibrium "), blue_bold(@sprintf("βₙ=%.2f",eq_betaN)), ")")
        pretty_table(
            data;
            formatters    = ft_printf("%5.3f", 2:4),
            header        = header,
            header_crayon = crayon"white bold",
            highlighters  = (hl1, hl2, hl3),
            tf            = tf_unicode_rounded
        )
    else
        @printf("\n(CNN model): Troyon Beta_N Limits\n")
        @printf("↳ (n=%d): βₙ=%.3f\n", CNN.n, CNN.βₙ_limit)
    end
end

function Check_validity_of_NN_for_given_input(TD::Troyon_Data, eqt::IMAS.equilibrium__time_slice; kwargs...)
    Check_validity_of_NN_for_given_input(TD.MLPs, eqt; kwargs...)
    Check_validity_of_NN_for_given_input(TD.CNN, eqt; kwargs...)
end

function Check_validity_of_NN_for_given_input(MLPs::Vector{MLP_Model}, eqt::IMAS.equilibrium__time_slice; kwargs...)
    verbose = get(kwargs, :verbose, false)

    # Calculate relevant physical parameters
    Aspect_Ratio = eqt.boundary.geometric_axis.r / eqt.boundary.minor_radius
    Elongation = eqt.boundary.elongation
    Triangularity = eqt.boundary.triangularity
    abs_q_min = minimum(abs.(eqt.profiles_1d.q))

    # Calculate PPF (Pressure Peaking Factor)
    PPF = eqt.profiles_1d.pressure[1] / Take_1D_average_over_volume(eqt, eqt.profiles_1d.pressure)

    # Internal inductance
    li = eqt.global_quantities.li_3

    # Define allowable ranges
    Aspect_Ratio_range = (1.3, 4.0)
    Elongation_range = (1.0, 2.3)
    abs_q_min_range = (1.05, 2.95)
    PPF_range = (1.5, 4.0)
    li_range = (0.5, 1.3)

    # Check each parameter for MLP NN
    MLP_params = Matrix{Any}(undef, 6, 5)
    MLP_params[1,:] .= check_parameter("R₀/a₀", Aspect_Ratio, Aspect_Ratio_range; model_name="MLP")
    MLP_params[2,:] .= check_parameter("Elongation", Elongation, Elongation_range; model_name="MLP")
    MLP_params[3,:] .= check_parameter("|q|_min", abs_q_min, abs_q_min_range; model_name="MLP")
    MLP_params[4,:] .= check_parameter("PPF", PPF, PPF_range; model_name="MLP")
    MLP_params[5,:] .= check_parameter("li", li, li_range; model_name="MLP")
    MLP_params[6,:] .= check_parameter_positivity("Triangularity", Triangularity; model_name="MLP")

    if verbose
        print_verbose_param_output(MLP_params; model_name="MLP")
    end
end


function Check_validity_of_NN_for_given_input(CNN::CNN_Model, eqt::IMAS.equilibrium__time_slice; kwargs...)
    verbose = get(kwargs, :verbose, false)

    # Calculate relevant physical parameters
    Aspect_Ratio = eqt.boundary.geometric_axis.r / eqt.boundary.minor_radius
    Elongation = eqt.boundary.elongation
    Triangularity = eqt.boundary.triangularity
    abs_q_min = minimum(abs.(eqt.profiles_1d.q))

    # Calculate PPF (Pressure Peaking Factor)
    PPF = eqt.profiles_1d.pressure[1] / Take_1D_average_over_volume(eqt, eqt.profiles_1d.pressure)

    # Internal inductance
    li = eqt.global_quantities.li_3

    # Check each parameter for CNN NN
    # CNN case (HL-2M tokamak: R0=178 cm, a0=65cm, R0/a0~2.74)
    CNN_params = Matrix{Any}(undef, 6, 5)
    CNN_params[1,:] .= check_parameter("R₀/a₀", Aspect_Ratio, (2.7, 2.8); model_name="CNN")
    CNN_params[2,:] .= check_parameter("Elongation", Elongation, (1.0, 1.833); model_name="CNN")
    CNN_params[3,:] .= check_parameter("Triangularity", Triangularity, (-0.6, 0.8); model_name="CNN")
    CNN_params[4,:] .= check_parameter("q_0", abs(eqt.global_quantities.q_axis), (1.155, 2.367); model_name="CNN")
    CNN_params[5,:] .= check_parameter("q_95", abs(eqt.global_quantities.q_95), (3.94, 8.207); model_name="CNN")
    CNN_params[6,:] .= check_parameter("q_min", abs_q_min, (1.146, 2.131); model_name="CNN")

    if verbose
        print_verbose_param_output(CNN_params; model_name="CNN")
    end
end

# Helper function to check parameter validity
function check_parameter(name::String, value::Float64, range::Tuple{Float64, Float64}; model_name::String="")
    lower, upper = range
    range_width = upper - lower
    pos_percentage = (value - lower) / range_width * 100

    if value < lower || value > upper
        @warn("($(model_name)): $name ($value) is outside the allowable range [$lower ~ $upper]")
        status = "Out of Range"
    else
        lower_edge = lower + 0.05 * range_width
        upper_edge = upper - 0.05 * range_width
        if value < lower_edge || value>upper_edge
            @info("($model_name): $name ($value) is near the boundary of the allowable range [$lower ~ $upper]")
            status = "Marginal"
        else
            status = "Okay"
        end
    end
    return (name, value, @sprintf("[%.2f ~ %.2f]",range[1], range[2]), @sprintf("%.f %%", pos_percentage), status)
end

function check_parameter_positivity(name::String, value::Float64; model_name::String="")
    if value>=0
        status = "Okay"
    else
        @warn("($model_name): $name ($value) is negative. Out of trained range")
        status = "Out of Range"
    end
    return (name, value,"positive (≥0)", "", status)
end

function print_verbose_param_output(data; model_name::String="")
    blue_bold = Crayon(foreground=:blue, bold=true)
    white_bold = Crayon(foreground=:white, bold=true)
    println(blue_bold("\n$(model_name):"),white_bold(" validity of equilibrium parameters"))
    header = ["param.", "value", "allowable range", "rel. pos", "status"];

    hl1 = Highlighter( (data, i, j) -> (j in (1,2,5)) && (data[i, end]=="Out of Range"), crayon"red bold");
    hl2 = Highlighter( (data, i, j) -> (j in (1,2,5)) && (data[i, end]=="Marginal"), crayon"yellow bold");
    hl3 = Highlighter( (data, i, j) -> (j==5) && (data[i, end]=="Okay"), crayon"green");
    pretty_table(
        data;
        formatters    = ft_printf("%5.2f", 2:4),
        header        = header,
        header_crayon = crayon"white bold",
        highlighters  = (hl1,hl2,hl3),
        tf            = tf_unicode_rounded
    )
end



function _calculate_MLP_neurons(TD::Troyon_Data, eqt::IMAS.equilibrium__time_slice)

    if (isempty(TD.sampPoints.R) || isempty(TD.sampPoints.q))
        Sample_points_from_equilibrium(TD, eqt)
    end

    # 19 neurons from RZ boundary points
    Xb = _convert_RZ_samples_into_19_normalized_neurons(TD,eqt)

    # 12 neurons from safety factor
    Xq = TD.sampPoints.q

    # 11 neurons from normalized pressure
    Xp = TD.sampPoints.pressure[2:end]/TD.sampPoints.pressure[1]

    # 42 neurons found from equilibrium
    X_neurons_from_eqt = [Xb; Xq; Xp]
    return X_neurons_from_eqt
end

function _convert_RZ_samples_into_19_normalized_neurons(TD::Troyon_Data, eqt::IMAS.equilibrium__time_slice)
    # Calculate R_hat and Z_hat (normalization)
    bdy = eqt.boundary
    R0 = bdy.geometric_axis.r
    Z0 = bdy.geometric_axis.z

    R_hat = TD.sampPoints.R ./ R0
    Z_hat = (TD.sampPoints.Z .- Z0) ./ R0

    # 19 neurons from boundary
    Xb = [R_hat[1:4]; Z_hat[5]; R_hat[6:12]; Z_hat[13]; R_hat[14:17]; Z_hat[17]; R_hat[18]]
end

function _set_CNN_input_neurons_from_sampled_points(TD::Troyon_Data,  eqt::IMAS.equilibrium__time_slice)

    Xb = _convert_RZ_samples_into_19_normalized_neurons(TD,eqt)
    Xp = TD.sampPoints.pressure[2:end]/TD.sampPoints.pressure[1]
    Xq = TD.sampPoints.q

    input_1 = Float32.(reshape(Xb,1,19,1)) # Boundary input
    input_2 = Float32.(reshape(Xp,1,11,1)) # Pressure
    input_3 = Float32.(reshape(Xq,1,12,1)) # Safety factor

    input_4 = Float32.(reshape([eqt.global_quantities.li_3],1,1)) # internal inductance

    # Calculate PPF (Pressure Peaking Factor)
    PPF = eqt.profiles_1d.pressure[1] / Take_1D_average_over_volume(eqt, eqt.profiles_1d.pressure)
    input_5 = Float32.(reshape([PPF],1,1))

    TD.CNN.input = Dict("input_1"=>input_1, "input_2"=>input_2, "input_3"=>input_3, "input_4"=>input_4, "input_5"=>input_5)
end

function Sample_points_from_equilibrium(TD::Troyon_Data, eqt::IMAS.equilibrium__time_slice)
    # 18 sample RZ points on a boundary
    TD.sampPoints.R, TD.sampPoints.Z =Sample_RZ_points_on_a_boundary(eqt);

    # Construct interpolators for 1D q and norm_pressure profiles
    # 12 sample points representing safety factor profile
    itp_q = interpolate(Float64.(eqt.profiles_1d.psi_norm), Float64.(eqt.profiles_1d.q), BSplineOrder(4))
    TD.sampPoints.q = abs.(itp_q.(TD.sampPoints.ψₙ)); # q must be positive

    # 12 sample points representing normalized pressure profile
    itp_p = interpolate(Float64.(eqt.profiles_1d.psi_norm), Float64.(eqt.profiles_1d.pressure), BSplineOrder(4))
    TD.sampPoints.pressure = itp_p.(TD.sampPoints.ψₙ);
end

function Sample_RZ_points_on_a_boundary(eqt::IMAS.equilibrium__time_slice)

    R0 = eqt.boundary.geometric_axis.r
    Z0 = eqt.boundary.geometric_axis.z

    bdy_R = Float64.(eqt.boundary.outline.r)
    bdy_Z = Float64.(eqt.boundary.outline.z)

    #  Calculate geometric angle
    theta = atan.(bdy_Z .- Z0, bdy_R .- R0)
    theta = mod.(theta, 2 * pi)

    # make theta unique
    unique_theta = unique(theta)
    I_unique = indexin(unique_theta, theta)
    theta = theta[I_unique]
    bdy_R = bdy_R[I_unique]
    bdy_Z = bdy_Z[I_unique]

    # sort theta, for interpolation later
    p = sortperm(theta)
    theta = theta[p]
    bdy_R = bdy_R[p]
    bdy_Z = bdy_Z[p]

    # Add ghost points  to both boundary considering periodic nature
    # This can prevent wrong extrapolation
    theta = [theta[end]-2π; theta; 2π+theta[1]]
    bdy_R = [bdy_R[end]; bdy_R; bdy_R[1]]
    bdy_Z = [bdy_Z[end]; bdy_Z; bdy_Z[1]]

    # Find sample boundary points for NN
    th_samp = Vector(0:15) * 22.5 / 180 * π

    itp_R = interpolate(theta, bdy_R, BSplineOrder(4))
    itp_Z = interpolate(theta, bdy_Z, BSplineOrder(4))

    R_samp = itp_R.(th_samp)
    Z_samp = itp_Z.(th_samp)

    Zmin, Imin = findmin(bdy_Z)
    Zmax, Imax = findmax(bdy_Z)

    R_samp = [R_samp; bdy_R[Imin]; bdy_R[Imax]]
    Z_samp = [Z_samp; Zmin; Zmax]

    return R_samp, Z_samp
end



function Take_1D_average_over_volume(eqt::IMAS.equilibrium__time_slice, target_1D_variable::Vector)
    # Check the length of given target_1D_variable
    if length(target_1D_variable) != length(eqt.profiles_1d.volume)
        @printf("Error: Length of target_1D_variable (%d) does not match length of eqt.profiles_1d.volume (%d).\n",
            length(target_1D_variable), length(eqt.profiles_1d.volume))
        throw(ArgumentError("The length of target_1D_variable does not match the length of eqt.profiles_1d.volume."))
    end

    var1D = target_1D_variable

    dV_dpsi = eqt.profiles_1d.dvolume_dpsi
    Δψ = diff(eqt.profiles_1d.psi)

    # Calculate the integral using the trapezoidal rule
    integral_over_volume = sum(0.5*(var1D[1:end-1].*dV_dpsi[1:end-1] .+ var1D[2:end].*dV_dpsi[2:end]).*Δψ)

    # Calculate the volume average
    average_over_volume = integral_over_volume / eqt.global_quantities.volume
    return average_over_volume
end


function plot_sample_points(TD::Troyon_Data, eqt::IMAS.equilibrium__time_slice; fileName_prefix::String = "Troyon_Sample_Points", file_type::String="png", title_prefix="")
    if (isempty(TD.sampPoints.R) || isempty(TD.sampPoints.q))
        Sample_points_from_equilibrium(TD, eqt)
    end

    # 1D plot
    plt_q=plot(
        eqt.profiles_1d.psi_norm, abs.(eqt.profiles_1d.q),
        color = :black,
        linestyle = :solid,
        linewidth = 2.5,
        label = "FUSE",
        title = "|Safety factor q|",
        xlabel = raw"$\psi_N$",
        grid = true,
        dpi=300
    )
    scatter!(plt_q,
        TD.sampPoints.ψₙ, TD.sampPoints.q;
        marker = (:circle, 8),
        markerstrokecolor = :red,
        markeralpha = 0.5,
        markerstrokewidth = 2,
        label = "12 samples"
    )
    hline!(plt_q, [1], linestyle=:dash, label="q=1")
    ylims!(0, ylims(plt_q)[2])


    plt_pressure=plot(
        eqt.profiles_1d.psi_norm, eqt.profiles_1d.pressure,
        color = :black,
        linestyle = :solid,
        linewidth = 2.5,
        label = "FUSE",
        xlabel = raw"$\psi_N$",
        ylabel = "(Pa)",
        title = "Pressure"
    )

    scatter!(plt_pressure,
        TD.sampPoints.ψₙ[2:end], TD.sampPoints.pressure[2:end],
        marker = (:circle, 8),
        markerstrokecolor = :red,
        markeralpha = 0.5,
        markerstrokewidth = 2,
        label = "11 samples"
    )

    # Plot Boundary (2D)
    bdy_R = eqt.boundary.outline.r
    bdy_Z = eqt.boundary.outline.z
    R0 = eqt.boundary.geometric_axis.r
    Z0 = eqt.boundary.geometric_axis.z


    plt_bdy=plot(
        [bdy_R; bdy_R[1]], [bdy_Z; bdy_Z[1]];
        color = :black,
        linestyle = :solid,
        linewidth = 2.5,
        label = "FUSE",
        aspect_ratio=:equal,
        legend = :outerbottom,
        title="Boundary",
        dpi=300
    )
    scatter!(plt_bdy,
        TD.sampPoints.R[1:end-2], TD.sampPoints.Z[1:end-2];
        marker = (:circle, 8),
        markerstrokecolor = :red,
        markeralpha = 0.5,
        markerstrokewidth = 2,
        label = "RZ (uniform angle)"
    )
    scatter!(plt_bdy,
        TD.sampPoints.R[end-1:end], TD.sampPoints.Z[end-1:end];
        marker = (:x, 8),
        color = :red,
        markerstrokewidth = 5,
        markeralpha = 0.5,
        label = "RZ (top & bottom)"
    )

    tmp_NR = length(TD.sampPoints.R) - 2

    RR_mat = hcat( fill(R0, tmp_NR), TD.sampPoints.R[1:end-2], fill(NaN, tmp_NR))'
    ZZ_mat = hcat( fill(Z0, tmp_NR), TD.sampPoints.Z[1:end-2], fill(NaN, tmp_NR))'

    plot!(plt_bdy,
        vec(RR_mat), vec(ZZ_mat);
        linestyle = :dash,
        color = RGB(0.5, 0.5, 0.5),
        linewidth = 0.5,
        label = ""
    )
    xlabel!("R (m)")
    ylabel!("Z (m)")

    # # Arrange plots
    my_layout = @layout [[a; b] c{0.5w}]

    if isempty(title_prefix)
        fig=plot(plt_q, plt_pressure, plt_bdy; layout =my_layout, size=(700,500), plot_title=@sprintf("time=%.2g s",eqt.time), plot_titlevspan=0.07)
    else
        fig=plot(plt_q, plt_pressure, plt_bdy; layout =my_layout, size=(700,500), plot_title=@sprintf("%s @ t=%.2g s",title_prefix, eqt.time), plot_titlevspan=0.07)
    end

    file_name =@sprintf("%s_t=%.2g_secs.%s",fileName_prefix, eqt.time, file_type)
    savefig(file_name)

    white_bold = Crayon(foreground=:yellow, bold=true)
    println("\nPlot of Sample_Points is saved into:", white_bold(file_name))
    display(fig)
end

end # module TroyonBetaNN
