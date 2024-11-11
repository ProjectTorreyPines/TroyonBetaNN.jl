function _print_results_to_stdout(TD; kwargs...)
    _print_results_to_stdout(TD.MLPs; kwargs...)
    return _print_results_to_stdout(TD.CNN; kwargs...)
end

function print_verbose_param_output(data; model_name::String="")
    header = ["param.", "value", "allowable range", "rel. pos", "status"]

    hl1 = Highlighter((data, i, j) -> (j in (1, 2, 4, 5)) && (data[i, end] == "Out of Range"), crayon"red bold")
    hl2 = Highlighter((data, i, j) -> (j in (1, 2, 4, 5)) && (data[i, end] == "Marginal"), crayon"yellow bold")
    hl3 = Highlighter((data, i, j) -> (j == 5) && (data[i, end] == "Okay"), crayon"green")

    magenta_bold = Crayon(; foreground=:magenta, bold=true)

    str_model_name = magenta_bold("[$model_name model]")
    return pretty_table(
        data;
        formatters=ft_printf("%5.2f", 2:4),
        header=header,
        header_crayon=crayon"white bold",
        highlighters=(hl1, hl2, hl3),
        tf=tf_unicode_rounded,
        title="\n$(str_model_name)\n validity of equilibrium parameters",
        title_alignment=:c,
        title_same_width_as_table=true
    )
end


function _print_results_to_stdout(MLPs::Vector{MLP_Model}; kwargs...)
    verbose = get(kwargs, :verbose, false)
    eq_betaN = get(kwargs, :eq_betaN, -1.0)

    MLP_stability_vec = Vector{String}(undef, length(MLPs))
    for (n, MLP) in pairs(MLPs)
        if (eq_betaN > MLP.βₙ_limit)
            MLP_stability_vec[n] = "Unstable"
        elseif eq_betaN > 0.95 * MLP.βₙ_limit
            MLP_stability_vec[n] = "Marginal"
        else
            MLP_stability_vec[n] = "Stable"
        end
    end

    if verbose && eq_betaN > 0
        blue_bold = Crayon(; foreground=:blue, bold=true)
        magenta_bold = Crayon(; foreground=:magenta, bold=true)

        header = ["Tor. mode", "Troyon βₙ Limit", "Stabiltiy"]

        data = hcat(getfield.(MLPs, :n), getfield.(MLPs, :βₙ_limit), MLP_stability_vec)

        hl1 = Highlighter((data, i, j) -> (j in (1, 3)) && (data[i, end] == "Unstable"), crayon"red bold")
        hl2 = Highlighter((data, i, j) -> (j in (1, 3)) && (data[i, end] == "Marginal"), crayon"yellow bold")
        hl3 = Highlighter((data, i, j) -> (j == 3) && (data[i, end] == "Stable"), crayon"green")

        model_name = magenta_bold("[MLP model]")
        str_eq_betaN = blue_bold(@sprintf("βₙ=%.2f", eq_betaN))
        pretty_table(
            data;
            formatters=ft_printf("%5.3f", 2:4),
            header=header,
            header_crayon=crayon"white bold",
            highlighters=(hl1, hl2, hl3),
            tf=tf_unicode_rounded,
            title="\n$model_name\n (Equilibrium $str_eq_betaN)",
            title_alignment=:c,
            title_same_width_as_table=true
        )
    else
        @printf("\n[MLP]: Troyon Beta_N Limits\n")
        for (n, this_MLP) in pairs(MLPs)
            @printf("  ↳ (n=%d): βₙ=%.3f (%s)\n", this_MLP.n, this_MLP.βₙ_limit, MLP_stability_vec[n])
        end
    end
end

function _print_results_to_stdout(CNN::CNN_Model; kwargs...)
    verbose = get(kwargs, :verbose, false)
    eq_betaN = get(kwargs, :eq_betaN, -1.0)

    if (eq_betaN > CNN.βₙ_limit)
        stability = "Unstable"
    elseif eq_betaN > 0.95 * CNN.βₙ_limit
        stability = "Marginal"
    else
        stability = "Stable"
    end

    if verbose && eq_betaN > 0
        blue_bold = Crayon(; foreground=:blue, bold=true)
        magenta_bold = Crayon(; foreground=:magenta, bold=true)

        header = ["Tor. mode", "Troyon βₙ Limit", "Stabiltiy"]


        data = hcat(CNN.n, CNN.βₙ_limit, stability)

        hl1 = Highlighter((data, i, j) -> (j in (1, 3)) && (data[i, end] == "Unstable"), crayon"red bold")
        hl2 = Highlighter((data, i, j) -> (j in (1, 3)) && (data[i, end] == "Marginal"), crayon"yellow bold")
        hl3 = Highlighter((data, i, j) -> (j == 3) && (data[i, end] == "Stable"), crayon"green")

        model_name = magenta_bold("[CNN model]")
        str_eq_betaN = blue_bold(@sprintf("βₙ=%.2f", eq_betaN))
        pretty_table(
            data;
            formatters=ft_printf("%5.3f", 2:4),
            header=header,
            header_crayon=crayon"white bold",
            highlighters=(hl1, hl2, hl3),
            tf=tf_unicode_rounded,
            title="\n$model_name\n (Equilibrium $str_eq_betaN)",
            title_alignment=:c,
            title_same_width_as_table=true
        )
    else
        @printf("\n[CNN]: Troyon Beta_N Limits\n")
        @printf("  ↳ (n=%d): βₙ=%.3f (%s)\n", CNN.n, CNN.βₙ_limit, stability)
    end
end
