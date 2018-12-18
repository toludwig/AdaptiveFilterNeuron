include("AdaptiveFilter.jl")

"""
This file contains the code for all the plots in the result section
of the thesis in the order in which they appear.
(`latencies` and `alpha_lambda` are not part of the thesis.)
For running, uncomment the desired method at the bottom.
"""


function pretest_sin()
    global training_cycles = 5000
    global task = "sin"
    global show = "out"
    global α = 2
    global λ = 2
    global η = 1e-4
    global latency = -.6

    ns = 2:2:16
    function mses_n(algo_code)
        global algo = algo_code

        mses = []
        for n_ in ns
            global n = n_
            io, tp, u0, tsave = init_training(task, training_cycles)
            sol      = run(io, u0, tsave)
            _, error = test_error(sol, io, tp)
            push!(mses, error)
        end
        return mses
    end
    af_mses = mses_n(:af)
    la_mses = mses_n(:lazar)
    n_plot = plot(ns, [af_mses, la_mses],
                  xlabel="n", ylabel="mse", xticks=ns,
                  labels=["AF", "Lazar"])
    scatter!([10, 10], [af_mses[5], la_mses[5]], lab="n=10")
    global n = 10 # reset

    rates = 2:2:16
    function mses_rate(algo_code)
        global show = "out" # dummy, only save end
        global algo = algo_code

        mses = []
        for rate_ in rates
            global rate = rate_
            global κ = (b+offset)/rate/θ   # capacitance (integrator weight)
            io, tp, u0, tsave = init_training(task, training_cycles)
            sol      = run(io, u0, tsave)
            _, error = test_error(sol, io, tp)
            push!(mses, error)
        end
        return mses
    end
    af_mses = mses_rate(:af)
    la_mses = mses_rate(:lazar)
    rate_plot = plot(rates, [af_mses, la_mses],
                     xlabel="rate", ylabel="mse", xticks=rates,
                     labels=["AF", "Lazar"])
    scatter!([10, 10], [af_mses[5], la_mses[5]], lab="rate=10")
    # reset
    global rate = 10
    global κ = (b+offset)/rate/θ   # capacitance (integrator weight)


    latencies = 0:.3:3
    function mses_lat(algo_code)
        global show = "out" # dummy, only save end
        global algo = algo_code

        mses = []
        for latency_ in latencies
            global latency = - latency_ # mind the minus!
            io, tp, u0, tsave = init_training(task, training_cycles)
            sol      = run(io, u0, tsave)
            _, error = test_error(sol, io, tp)
            push!(mses, error)
        end
        return mses
    end
    af_mses = mses_lat(:af)
    la_mses = mses_lat(:lazar)
    lat_plot = plot(latencies, [af_mses, la_mses],
                    xlabel="latency", ylabel="mse", xticks=0:.6:3,
    labels=["AF", "Lazar"])
    scatter!([.6, .6], [af_mses[3], la_mses[3]], lab="latency=0.6")
    global latency = -.6 # reset


    alphas = .25:.25:3.25
    function mses_algo(algo_code)
        global algo = algo_code

        mses = []
        for alpha_ in alphas
            global α = alpha_
            global λ = alpha_
            io, tp, u0, tsave = init_training(task, training_cycles)
            sol      = run(io, u0, tsave)
            _, error = test_error(sol, io, tp)
            push!(mses, error)
        end
        return mses
    end
    af_mses = mses_algo(:af)
    la_mses = mses_algo(:lazar)
    alpha_plot = plot(alphas, [af_mses, la_mses],
                      xlabel="alpha, lambda", ylabel="mse", xticks=.5:.5:3,
                      labels=["AF", "Lazar"])
    scatter!([2, 2], [af_mses[8], la_mses[8]], lab="alpha=lambda=2")
    # reset
    global α = 2
    global λ = 2


    plot(n_plot, rate_plot, lat_plot, alpha_plot,
         title=["A" "B" "C" "D"], titleloc=:left,
         layout=(2,2), size=(500, 500))
    savefig("../../Thesis/figures/pretest-alpha$α-latency$latency.pdf")
end

function latencies()
    global training_cycles = 5000
    global task = "sin"
    global show = "out"
    global α = 2
    global λ = 2
    global η = 1e-4
    latencies = 0:.3:3.
    function mses_lat(algo_code)
        global algo = algo_code

        mses = []
        for latency_ in latencies
            global latency = - latency_ # mind the minus!
            io, tp, u0, tsave = init_training(task, training_cycles)
            sol      = run(io, u0, tsave)
            _, error = test_error(sol, io, tp)
            push!(mses, error)
        end
        return mses
    end
    af_mses = mses_lat(:af)
    la_mses = mses_lat(:lazar)
    n_plot = plot(latencies, [af_mses, la_mses],
                  xlabel="latency", ylabel="mse", xticks=0:.6:3,
                  labels=["AF", "Lazar"])
    scatter!([.6, .6], [af_mses[3], la_mses[3]], lab="latency=.6")
    savefig("../../Thesis/figures/latencies.pdf")
end

function alpha_lambda()
    global training_cycles = 5000
    global task = "sin"
    global show = "out"
    global latency = -.6
    global η = 1e-4
    alphas = .25:.25:3.5
    function mses_algo(algo_code)
        global algo = algo_code

        mses = []
        for alpha_ in alphas
            global α = alpha_
            global λ = alpha_
            io, tp, u0, tsave = init_training(task, training_cycles)
            sol      = run(io, u0, tsave)
            _, error = test_error(sol, io, tp)
            push!(mses, error)
        end
        return mses
    end
    af_mses = mses_algo(:af)
    la_mses = mses_algo(:lazar)
    alpha_plot = plot(alphas, [af_mses, la_mses],
                      xlabel="alpha, lambda", ylabel="mse", xticks=alphas,
                      labels=["AF", "Lazar"],
                      size=(300, 200))
    savefig("../../Thesis/figures/alpha-lambda-latency$latency.pdf")
end

function training_error_sin()
    global training_cycles = 10000
    global task = "sin"
    global α = 2
    global λ = 2
    global η = 1e-4
    global latency = -.6
    function plot_algo(algo_code, algo_name, point_names)
        global algo = algo_code
        global show = "mse_evol" # to save all periods
        io, tp, u0, tsave = init_training(task, training_cycles)
        sol    = run(io, u0, tsave)

        p1, p2 = (2500, 10000)
        pn1, pn2 = point_names
        mses, sols = running_mse(sol, io, [p1, p2])

        al_evol = plot(mses, title="$algo_name", lab="MSE",
                       titleloc=:left, leg=false,
                       ylims=(0,.8),
                       ylabel="mse", xlabel="periods")
        scatter!([p1, p2], [mses[p1], mses[p2]])
                 #series_annotations = [pn1, pn2])
        annotate!([(p1, mses[p1], text(pn1, :bottom)),
                   (p2, mses[p2], text(pn2, :bottom))])
        al_out1 = plot([t->out(sols[1], t), io.z], sols[1].t,
                       xticks=sols[1].t[1]:tp:sols[1].t[end],
                       title="$pn1 (mse: $(mses[p1]))",
                       titleloc=:left, leg=false)
        al_out2 = plot([t->out(sols[2], t), io.z], sols[2].t,
                       xticks=sols[2].t[1]:tp:sols[2].t[end],
                       title="$pn2 (mse: $(mses[p2]))",
                       titleloc=:left, leg=false,  xlabel="time")

        l = @layout [ a{.5w} [b{.5h}
                              c{.5h}] ]
        plot(al_evol, al_out1, al_out2, layout=l)
    end
    af_plot = plot_algo(:af, "AF", ("A", "B"))
    la_plot = plot_algo(:lazar, "Lazar", ("C", "D"))
    plot(af_plot, la_plot, layout=(2,1), size=(500, 500))
    savefig("../../Thesis/figures/sin-training-cycles20000.pdf")
end

function weights_taps_sin()
    global training_cycles = 10000
    global task = "sin"
    global α = 2
    global λ = 2
    global η = 1e-4
    global latency = -.6
    function plot_algo(algo_code, algo_name)
        global show = "w_evol"
        global algo = algo_code
        io, tp, u0, tsave = init_training(task, training_cycles)
        sol    = run(io, u0, tsave)
        w_plot = evol_plot("w_evol", sol, io)
        # sol can be used to plot y_end as well
        y_plot = end_plot("y", sol, io, tp)
        # stack w and y horizontally
        return (w_plot, y_plot)
    end
    # stack af and lazar vertically
    af_w, af_y = plot_algo(:af, "AF")
    la_w, la_y = plot_algo(:lazar, "Lazar")
    plot(af_w, af_y, la_w, la_y, layout=(2,2), leg=false)
    savefig("../../Thesis/figures/sin-weights-taps.pdf")
end

function imp_sin()
    global training_cycles = 10000
    global task = "sin"
    global α = 2
    global λ = 2
    global η = 1e-4
    global latency = -.6

    global algo = :af
    io, tp, u0, tsave = init_training(task, training_cycles)
    sol    = run(io, u0, tsave)
    af_imp = impulse_response(sol)

    global algo = :lazar
    io, tp, u0, tsave = init_training(task, training_cycles)
    sol    = run(io, u0, tsave)
    la_imp = impulse_response(sol)

    af_impi = plot([y_i(af_imp, i) for i in 1:n],
                   af_imp.t, leg=false,
                   labels=["y$i" for i in 1:n],
                   title="AF, individual IR",
                   titleloc=:left, legend=:bottomright)
    la_impi = plot([y_i(la_imp, i) for i in 1:n],
                   la_imp.t,
                   labels=["y$i" for i in 1:n],
                   title="Lazar, individual IR",
                   titleloc=:left, legend=:bottomright)
    both = plot([t->out(af_imp, t), t->out(la_imp, t)],
                af_imp.t,
                label=["AF", "Lazar"], xlabel="time",
                title="Total IR",
                titleloc=:left, legend=:topright)

    plot(af_impi, la_impi, both, layout=(3,1), size=(400,600))

    savefig("../../Thesis/figures/sin-imp.pdf")
end

function task_noise()
    global training_cycles = 10000
    global task = "noise"
    global show = "out"
    global α = 2
    global λ = 2
    global η = 1e-4
    global latency = -.6
    nstds = 0:.3:1.8
    function mses_algo(algo_code)
        global algo = algo_code

        mses = []
        for nstd in nstds
            global noise_std = nstd
            io, tp, u0, tsave = init_training(task, training_cycles)
            sol    = run(io, u0, tsave)
            _, error = test_error(sol, io, tp)
            push!(mses, error)
        end
        return mses
    end
    af_mses = mses_algo(:af)
    la_mses = mses_algo(:lazar)
    main = plot(nstds, [af_mses, la_mses],
                title="Noise task", titleloc=:left,
                xlabel="noise std", ylabel="mse",
                labels=["AF", "Lazar"], leg=:topleft)
    # again for subplots
    global noise_std = 1.2
    global algo = :lazar
    io, tp, u0, tsave = init_training(task, training_cycles)
    sol    = run(io, u0, tsave)
    la_sol, error = test_error(sol, io, tp)
    global algo = :af
    io, tp, u0, tsave = init_training(task, training_cycles)
    sol    = run(io, u0, tsave)
    af_sol, error = test_error(sol, io, tp)

    sub1 = plot([io.x, io.z], la_sol.t, labels=["input", "target"],
                title="for std = 1.2", titleloc=:left)
    sub2 = plot([t->out(la_sol, t), io.z], la_sol.t,
                title="Lazar", titleloc=:left, leg=false)
    sub3 = plot([t->out(af_sol, t), io.z], af_sol.t,
                title="AF", titleloc=:left, leg=false)
    l = @layout [ a{.4w} [b{.33h}
                          c{.33h}
                          d{.33h}] ]
    plot(main, sub1, sub2, sub3, layout=l)
    savefig("../../Thesis/figures/noise-std-sub.pdf")
end

function task_mix3()
    global training_cycles = 25000
    global test_cycles = 1 # because 1 cycle is already very long
    global task = "mix3"
    global show = "out"
    global α = 3 # mix3 specific
    global λ = 3
    global η = 1e-3
    global latency = -.6
    io, _ = init_task("1of3") # NOTE important for io in plot
    function out_algo(algo_code)
        global algo = algo_code
        global mode = "train"
        io, tp, u0, tsave = init_training(task, training_cycles)
        sol    = run(io, u0, tsave)
        sol2, error = test_error(sol, io, tp)
        return sol2, error
    end
    sol_af, mse_af = out_algo(:af)
    sol_la, mse_la = out_algo(:lazar)
    plot([t->io.z(t), t->out(sol_af, t), t->out(sol_la, t)], sol_af.t,
         labels=["target", "AF (MSE: $mse_af)", "Lazar (MSE: $mse_la)"],
         title="mix3", xlabel="time")
    savefig("../../Thesis/figures/mix3-task.pdf")
    global test_cycles = 3 # reset
end

function task_1of3()
    global training_cycles = 25000
    global task = "1of3"
    global show = "out"
    global α = 3
    global λ = 3
    global η = 1e-3
    global latency = -1 # tp/4 = 4/4 = 1
    io, _ = init_task("1of3") # NOTE important for io in plot
    function out_algo(algo_code)
        global algo = algo_code
        io, tp, u0, tsave = init_training(task, training_cycles)
        sol         = run(io, u0, tsave)
        sol2, error = test_error(sol, io, tp)
        return sol2, error
    end
    sol_af, mse_af = out_algo(:af)
    sol_la, mse_la = out_algo(:lazar)
    plot([io.x, io.z, t->out(sol_af, t), t->out(sol_la, t)], sol_af.t,
         labels=["input", "target",
                 "AF (MSE: $mse_af)", "Lazar (MSE: $mse_la)"],
         title="1of3", xlabel="time")
    savefig("../../Thesis/figures/1of3-task.pdf")
end

function imp_1of3()
    global training_cycles = 25000
    global task = "1of3"
    global show = "out"
    global α = 3
    global λ = 3
    global latency = -1 # tp/4 = 4/4 = 1
    global η = 1e-3

    global algo = :af
    io, tp, u0, tsave = init_training(task, training_cycles)
    sol    = run(io, u0, tsave)
    af_imp = impulse_response(sol)

    global algo = :lazar
    io, tp, u0, tsave = init_training(task, training_cycles)
    sol    = run(io, u0, tsave)
    la_imp = impulse_response(sol)

    plot([t->out(af_imp, t), t->out(la_imp, t)],
         af_imp.t,
         label=["AF", "Lazar"], xlabel="time",
         title="Total impulse response",
         titleloc=:left, legend=:topright)
    savefig("../../Thesis/figures/imp-1of3.pdf")
end

function imp_mix3()
    global training_cycles = 1000
    global task = "mix3"
    global show = "out"
    global α = 3
    global λ = 3
    global latency = -.6 # tp/4 = 4/4 = 1
    global η = 1e-3

    global algo = :af
    io, tp, u0, tsave = init_training(task, training_cycles)
    sol    = run(io, u0, tsave)
    af_imp = impulse_response(sol)

    global algo = :lazar
    io, tp, u0, tsave = init_training(task, training_cycles)
    sol    = run(io, u0, tsave)
    la_imp = impulse_response(sol)

    plot([t->out(af_imp, t), t->out(la_imp, t)],
         af_imp.t,
         label=["AF", "Lazar"], xlabel="time",
         title="Total impulse response",
         titleloc=:left, legend=:topright)
    savefig("../../Thesis/figures/imp-mix3.pdf")
end

function task_relu()
    global training_cycles = 25000
    global task = "relu"
    global show = "out"
    global α = 2
    global λ = 2
    global η = 1e-4
    global latency = -.6
    #global offset = 1
    io, _ = init_task("relu") # NOTE important for io in plot
    function out_algo(algo_code)
        global algo = algo_code
        io, tp, u0, tsave = init_training(task, training_cycles)
        sol    = run(io, u0, tsave)
        sol2, error = test_error(sol, io, tp)
        return sol2, error
    end
    sol_af, mse_af = out_algo(:af)
    sol_naf, mse_naf = out_algo(:naf)
    sol_la, mse_la = out_algo(:lazar)
    plot([io.z, t->out(sol_af, t), t->r(sol_naf, t), # mind the r (ramp)
          t->out(sol_la, t)], sol_af.t,
         labels=["target", "AF (MSE: $mse_af)", "nAF (MSE: $mse_naf)",
                 "Lazar (MSE: $mse_la)"],
         title="reLU", xlabel="time")
    savefig("../../Thesis/figures/relu-task.pdf")
end

# Pretests
#pretest_sin()
#latencies()
#alpha_lambda()

# Sine recovery
#training_error_sin()
#weights_taps_sin()
#imp_sin()

# Other tasks
#task_noise()
#task_mix3()
#task_1of3()
task_relu()

#imp_1of3()
#imp_mix3()
