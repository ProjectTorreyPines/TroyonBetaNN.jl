function load_predefined_Troyon_NN_Models(; MLP_fileName::String="MLP_Model.json", CNN_fileName::String="CNN_Model.onnx")
    # Read MLP file
    MLP_file_path = joinpath(@__DIR__, "..", "data", MLP_fileName)
    data_from_file = JSON.parsefile(MLP_file_path)

    target_n_modes = [1, 2, 3]

    MLPs = Vector{MLP_Model}(undef, 3)
    for n in target_n_modes
        w_data = data_from_file[n]["W"]
        w_data = Float64.(hcat(w_data...)')

        v_data = Float64.(data_from_file[n]["V"])

        # Create MLP instance
        MLPs[n] = MLP_Model(n, w_data, v_data, NaN, MLP_file_path)
    end

    # Read CNN file
    CNN_file_path = joinpath(@__DIR__, "..", "data", CNN_fileName)
    CNN = CNN_Model(; model=ORT.load_inference(CNN_file_path), filePath=CNN_file_path)

    return Troyon_Data(Sample_Points(), MLPs, CNN)
end

function calculate_Troyon_beta_limits_for_IMAS_dd(dd::IMAS.dd; kwargs...)
    Neqt = length(dd.equilibrium.time_slice)
    TD_vec = [load_predefined_Troyon_NN_Models() for _ in 1:Neqt]

    calculate_Troyon_beta_limits_for_IMAS_dd(TD_vec, dd; kwargs...)
    return TD_vec
end

function calculate_Troyon_beta_limits_for_IMAS_dd(TD_vec::Vector{Troyon_Data}, dd::IMAS.dd; kwargs...)
    verbose = get(kwargs, :verbose, false)

    yellow_bold = Crayon(; foreground=:yellow, bold=true)
    for tid in 1:length(dd.equilibrium.time_slice)
        this_eqt = dd.equilibrium.time_slice[tid]

        if isnan(this_eqt.global_quantities.vacuum_toroidal_field.b0)
            @warn(@sprintf("Equilibrium time_slice #%d has no equilibrium information\nSkipping Troyon βₙ calculations ...\n", tid))
        else
            if verbose
                println(yellow_bold(@sprintf("\nFor equilibrium time_slice #%d @ t=%.2g secs", tid, this_eqt.time)))
            end
            calculate_Troyon_beta_limits_for_a_given_time_slice(TD_vec[tid], this_eqt; kwargs...)
        end
    end

    return TD_vec
end

function calculate_Troyon_beta_limits_for_a_given_time_slice(eqt::IMAS.equilibrium__time_slice; kwargs...)
    TD = load_predefined_Troyon_NN_Models()
    calculate_Troyon_beta_limits_for_a_given_time_slice(TD, eqt; kwargs...)
    return TD
end

function calculate_Troyon_beta_limits_for_a_given_time_slice(TD::Troyon_Data, eqt::IMAS.equilibrium__time_slice; kwargs...)
    silence = get(kwargs, :silence, false)
    verbose = get(kwargs, :verbose, false)

    if isnan(eqt.global_quantities.vacuum_toroidal_field.b0)
        @warn("Given time_slice has no equilibrium information\nSkipping Troyon βₙ calculations ...\n")

        # reset NN models' betaN value to NaN
        setfield!.(TD.MLPs, :βₙ_limit, NaN)
        TD.CNN.βₙ_limit = NaN
        return
    end

    check_validity_of_NN_for_given_input(TD, eqt; kwargs...)

    sample_points_from_equilibrium(TD, eqt)

    # First, MLP model
    # Calculate 42 neurons from sample Points on equilibrium
    X_neurons = _calculate_MLP_neurons(TD, eqt)

    # Calculate Troyon beta_N limits using MLP model
    for MLP in TD.MLPs
        X = [X_neurons; 1] # add a bias neuron (total 43 neurons)

        # activate hidden neurons
        Y = 1.0 ./ (1.0 .+ exp.(-MLP.W * X))
        Y = [Y; 1] # add a bias neuron

        MLP.βₙ_limit = (MLP.V') * Y
    end

    # Calculate Troyon beta_N limits (n=1) using CNN model
    _set_CNN_input_neurons_from_sampled_points(TD, eqt)
    CNN_output = TD.CNN.model(TD.CNN.input)["tf.math.multiply"]
    TD.CNN.βₙ_limit = Float64.(vec(CNN_output)[1])

    if ~silence
        equilibrium_βₙ = eqt.global_quantities.beta_normal
        _print_results_to_stdout(TD; eq_betaN=equilibrium_βₙ, kwargs...)

        if verbose
            plot_sample_points(TD, eqt; file_type="png")
        end
    end

    return TD
end

function check_validity_of_NN_for_given_input(TD::Troyon_Data, eqt::IMAS.equilibrium__time_slice; kwargs...)
    check_validity_of_NN_for_given_input(TD.MLPs, eqt; kwargs...)
    return check_validity_of_NN_for_given_input(TD.CNN, eqt; kwargs...)
end

function check_validity_of_NN_for_given_input(MLPs::Vector{MLP_Model}, eqt::IMAS.equilibrium__time_slice; kwargs...)
    verbose = get(kwargs, :verbose, false)

    # Calculate relevant physical parameters
    Aspect_Ratio = eqt.boundary.geometric_axis.r / eqt.boundary.minor_radius
    Elongation = eqt.boundary.elongation
    Triangularity = eqt.boundary.triangularity
    abs_q_min = minimum(abs.(eqt.profiles_1d.q))

    # Calculate PPF (Pressure Peaking Factor)
    PPF = eqt.profiles_1d.pressure[1] / take_1D_average_over_volume(eqt, eqt.profiles_1d.pressure)

    # Internal inductance
    li = eqt.global_quantities.li_3

    # Check each parameter for MLP NN
    MLP_params = Matrix{Any}(undef, 6, 5)
    MLP_params[1, :] .= check_parameter("R₀/a₀", Aspect_Ratio, (1.3, 4.0); kwargs..., model_name="MLP")
    MLP_params[2, :] .= check_parameter("Elongation", Elongation, (1.0, 2.3); kwargs..., model_name="MLP")
    MLP_params[3, :] .= check_parameter("|q|_min", abs_q_min, (1.05, 2.95); kwargs..., model_name="MLP")
    MLP_params[4, :] .= check_parameter("PPF", PPF, (1.5, 4.0); kwargs..., model_name="MLP")
    MLP_params[5, :] .= check_parameter("li", li, (0.5, 1.3); kwargs..., model_name="MLP")
    MLP_params[6, :] .= check_parameter_positivity("Triangularity", Triangularity; kwargs..., model_name="MLP")

    if verbose
        print_verbose_param_output(MLP_params; model_name="MLP")
    end
end


function check_validity_of_NN_for_given_input(CNN::CNN_Model, eqt::IMAS.equilibrium__time_slice; kwargs...)
    verbose = get(kwargs, :verbose, false)

    # Calculate relevant physical parameters
    Aspect_Ratio = eqt.boundary.geometric_axis.r / eqt.boundary.minor_radius
    Elongation = eqt.boundary.elongation
    Triangularity = eqt.boundary.triangularity
    abs_q_min = minimum(abs.(eqt.profiles_1d.q))

    # Calculate PPF (Pressure Peaking Factor)
    PPF = eqt.profiles_1d.pressure[1] / take_1D_average_over_volume(eqt, eqt.profiles_1d.pressure)

    # Internal inductance
    li = eqt.global_quantities.li_3

    # Check each parameter for CNN NN
    # CNN case (HL-2M tokamak: R0=178 cm, a0=65cm, R0/a0~2.74)
    CNN_params = Matrix{Any}(undef, 6, 5)
    CNN_params[1, :] .= check_parameter("R₀/a₀", Aspect_Ratio, (2.7, 2.8); kwargs..., model_name="CNN")
    CNN_params[2, :] .= check_parameter("Elongation", Elongation, (1.0, 1.833); kwargs..., model_name="CNN")
    CNN_params[3, :] .= check_parameter("Triangularity", Triangularity, (-0.6, 0.8); kwargs..., model_name="CNN")
    CNN_params[4, :] .= check_parameter("q_0", abs(eqt.global_quantities.q_axis), (1.155, 2.367); kwargs..., model_name="CNN")
    CNN_params[5, :] .= check_parameter("q_95", abs(eqt.global_quantities.q_95), (3.94, 8.207); kwargs..., model_name="CNN")
    CNN_params[6, :] .= check_parameter("q_min", abs_q_min, (1.146, 2.131); kwargs..., model_name="CNN")

    if verbose
        print_verbose_param_output(CNN_params; model_name="CNN")
    end
end

# Helper function to check parameter validity
function check_parameter(name::String, value::Float64, range::Tuple{Float64,Float64}; kwargs...)
    verbose = get(kwargs, :verbose, false)
    silence = get(kwargs, :silence, false)

    model_name = get(kwargs, :model_name, "")

    lower, upper = range
    range_width = upper - lower
    pos_percentage = (value - lower) / range_width * 100

    if value < lower || value > upper
        if ~verbose && ~silence
            @warn("[$(model_name)]: $name " * @sprintf("(%.3f)", value) * " is outside the limit [$lower ~ $upper]")
        end
        status = "Out of Range"
    else
        lower_edge = lower + 0.05 * range_width
        upper_edge = upper - 0.05 * range_width
        if value < lower_edge || value > upper_edge
            if ~verbose && ~silence
                @info("[$model_name]: $name " * @sprintf("(%.3f)", value) * " is too close to the limit [$lower, $upper]")
            end
            status = "Marginal"
        else
            status = "Okay"
        end
    end
    return (name, value, @sprintf("[%.2f ~ %.2f]", range[1], range[2]), @sprintf("%.f %%", pos_percentage), status)
end

function check_parameter_positivity(name::String, value::Float64; kwargs...)
    verbose = get(kwargs, :verbose, false)
    silence = get(kwargs, :silence, false)

    model_name = get(kwargs, :model_name, "")

    if value >= 0
        status = "Okay"
    else
        if ~verbose && ~silence
            @warn("[$model_name]: $name " * @sprintf("(%.3f)", value) * " is negative. Out of trained range")
        end
        status = "Out of Range"
    end
    return (name, value, "positive (≥0)", "", status)
end


function _calculate_MLP_neurons(TD::Troyon_Data, eqt::IMAS.equilibrium__time_slice)
    if (isempty(TD.sampPoints.R) || isempty(TD.sampPoints.q))
        sample_points_from_equilibrium(TD, eqt)
    end

    # 19 neurons from RZ boundary points
    Xb = _convert_RZ_samples_into_19_normalized_neurons(TD, eqt)

    # 12 neurons from safety factor
    Xq = TD.sampPoints.q

    # 11 neurons from normalized pressure
    Xp = TD.sampPoints.pressure[2:end] / TD.sampPoints.pressure[1]

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
    return Xb = [R_hat[1:4]; Z_hat[5]; R_hat[6:12]; Z_hat[13]; R_hat[14:17]; Z_hat[17]; R_hat[18]]
end

function _set_CNN_input_neurons_from_sampled_points(TD::Troyon_Data, eqt::IMAS.equilibrium__time_slice)
    Xb = _convert_RZ_samples_into_19_normalized_neurons(TD, eqt)
    Xp = TD.sampPoints.pressure[2:end] / TD.sampPoints.pressure[1]
    Xq = TD.sampPoints.q

    input_1 = Float32.(reshape(Xb, 1, 19, 1)) # Boundary input
    input_2 = Float32.(reshape(Xq, 1, 12, 1)) # Safety factor
    input_3 = Float32.(reshape(Xp, 1, 11, 1)) # Pressure

    input_4 = Float32.(reshape([eqt.global_quantities.li_3], 1, 1)) # internal inductance

    # Calculate PPF (Pressure Peaking Factor)
    PPF = eqt.profiles_1d.pressure[1] / take_1D_average_over_volume(eqt, eqt.profiles_1d.pressure)
    input_5 = Float32.(reshape([PPF], 1, 1))

    return TD.CNN.input = Dict("input_1" => input_1, "input_2" => input_2, "input_3" => input_3, "input_4" => input_4, "input_5" => input_5)
end

function sample_points_from_equilibrium(TD::Troyon_Data, eqt::IMAS.equilibrium__time_slice)
    # 18 sample RZ points on a boundary
    TD.sampPoints.R, TD.sampPoints.Z = sample_RZ_points_on_a_boundary(eqt)

    # Construct interpolators for 1D q and norm_pressure profiles
    # 12 sample points representing safety factor profile
    itp_q = interpolate(Float64.(eqt.profiles_1d.psi_norm), Float64.(eqt.profiles_1d.q), BSplineOrder(4))
    TD.sampPoints.q = abs.(itp_q.(TD.sampPoints.ψₙ)) # q must be positive

    # 12 sample points representing normalized pressure profile
    itp_p = interpolate(Float64.(eqt.profiles_1d.psi_norm), Float64.(eqt.profiles_1d.pressure), BSplineOrder(4))
    return TD.sampPoints.pressure = itp_p.(TD.sampPoints.ψₙ)
end

function sample_RZ_points_on_a_boundary(eqt::IMAS.equilibrium__time_slice)
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
    theta = [theta[end] - 2π; theta; 2π + theta[1]]
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

function take_1D_average_over_volume(eqt::IMAS.equilibrium__time_slice, target_1D_variable::Vector)
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
    integral_over_volume = sum(0.5 * (var1D[1:end-1] .* dV_dpsi[1:end-1] .+ var1D[2:end] .* dV_dpsi[2:end]) .* Δψ)

    # Calculate the volume average
    average_over_volume = integral_over_volume / eqt.global_quantities.volume
    return average_over_volume
end
