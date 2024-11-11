function plot_sample_points(TD::Troyon_Data, eqt::IMAS.equilibrium__time_slice; fileName_prefix::String="Troyon_Sample_Points", file_type::String="png", title_prefix="")
    if (isempty(TD.sampPoints.R) || isempty(TD.sampPoints.q))
        sample_points_from_equilibrium(TD, eqt)
    end

    # 1D plot
    plt_q = plot(
        eqt.profiles_1d.psi_norm, abs.(eqt.profiles_1d.q);
        color=:black,
        linestyle=:solid,
        linewidth=2.5,
        label="FUSE",
        title="|Safety factor q|",
        xlabel=raw"$\psi_N$",
        grid=true,
        dpi=300
    )
    scatter!(plt_q,
        TD.sampPoints.ψₙ, TD.sampPoints.q;
        marker=(:circle, 8),
        markerstrokecolor=:red,
        markeralpha=0.5,
        markerstrokewidth=2,
        label="12 samples"
    )
    hline!(plt_q, [1]; linestyle=:dash, label="q=1")
    ylims!(0, ylims(plt_q)[2])


    plt_pressure = plot(
        eqt.profiles_1d.psi_norm, eqt.profiles_1d.pressure;
        color=:black,
        linestyle=:solid,
        linewidth=2.5,
        label="FUSE",
        xlabel=raw"$\psi_N$",
        ylabel="(Pa)",
        title="Pressure"
    )

    scatter!(plt_pressure,
        TD.sampPoints.ψₙ[2:end], TD.sampPoints.pressure[2:end];
        marker=(:circle, 8),
        markerstrokecolor=:red,
        markeralpha=0.5,
        markerstrokewidth=2,
        label="11 samples"
    )

    # Plot Boundary (2D)
    bdy_R = eqt.boundary.outline.r
    bdy_Z = eqt.boundary.outline.z
    R0 = eqt.boundary.geometric_axis.r
    Z0 = eqt.boundary.geometric_axis.z


    plt_bdy = plot(
        [bdy_R; bdy_R[1]], [bdy_Z; bdy_Z[1]];
        color=:black,
        linestyle=:solid,
        linewidth=2.5,
        label="FUSE",
        aspect_ratio=:equal,
        legend=:outerbottom,
        title="Boundary",
        dpi=300
    )
    scatter!(plt_bdy,
        TD.sampPoints.R[1:end-2], TD.sampPoints.Z[1:end-2];
        marker=(:circle, 8),
        markerstrokecolor=:red,
        markeralpha=0.5,
        markerstrokewidth=2,
        label="RZ (uniform angle)"
    )
    scatter!(plt_bdy,
        TD.sampPoints.R[end-1:end], TD.sampPoints.Z[end-1:end];
        marker=(:x, 8),
        color=:red,
        markerstrokewidth=5,
        markeralpha=0.5,
        label="RZ (top & bottom)"
    )

    tmp_NR = length(TD.sampPoints.R) - 2

    RR_mat = hcat(fill(R0, tmp_NR), TD.sampPoints.R[1:end-2], fill(NaN, tmp_NR))'
    ZZ_mat = hcat(fill(Z0, tmp_NR), TD.sampPoints.Z[1:end-2], fill(NaN, tmp_NR))'

    plot!(plt_bdy,
        vec(RR_mat), vec(ZZ_mat);
        linestyle=:dash,
        color=RGB(0.5, 0.5, 0.5),
        linewidth=0.5,
        label=""
    )
    xlabel!("R (m)")
    ylabel!("Z (m)")

    # # Arrange plots
    my_layout = @layout [[a; b] c{0.5w}]

    if isempty(title_prefix)
        fig = plot(plt_q, plt_pressure, plt_bdy; layout=my_layout, size=(700, 500), plot_title=@sprintf("time=%.2g s", eqt.time), plot_titlevspan=0.07)
    else
        fig = plot(plt_q, plt_pressure, plt_bdy; layout=my_layout, size=(700, 500), plot_title=@sprintf("%s @ t=%.2g s", title_prefix, eqt.time), plot_titlevspan=0.07)
    end

    file_name = @sprintf("%s_t=%.2g_secs.%s", fileName_prefix, eqt.time, file_type)
    savefig(file_name)

    white_bold = Crayon(; foreground=:yellow, bold=true)
    println("\nPlot of Sample_Points is saved into:", white_bold(file_name))
    return display(fig)
end
