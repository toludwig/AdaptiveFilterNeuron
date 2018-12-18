using DifferentialEquations, Plots
import StatsBase, JSON, Random

"""
This file contains the definition of the Adaptive Filter model as well as
the Lazar model as an ODE to be solved by the DifferentialEquations library.
"""

#=== meta execution ========#
exec = false            # if false, no solving or plotting from this file
algo = :naf             # options: :af, :naf, :lazar
task = "relu"           # options: recovery:     "sin", "mix2", "mix3",
                        # linear computation:    "1of3", "shift", "noise"
                        # nonlinear computation: "abs", "relu", "sign"
mode = "train"          # options: "train", "test" (disables learning rule)
show = "mse_evol"       # options: "out", "y", "y*w", "zyw", "imp", "impi",
                        # "w_evol", "mse_evol" (measured during training)
mark = ""               # any mark to put in the figure filename

#=== input/output setup ====#
lp = 3                  # lowest = shortest period (fastest frequency)
mp = 4                  # middle period (for mix3 and 1of3 task)
hp = 5                  # highest = longest period (slowest frequency)
phase = lp/4            # phase for task="shift" (additional to latency)
offset = 0              # y-offset of signal (important for nonlinearity)
latency = -.60          # accounts for latency of information due to spikes
noise_std = .5          # std for gaussian white noise

#=== encoder parameters ====#
rate = 10               # the resting firing rate (spikes/sec)
b = 3.0                 # bias (> |minimal input|, s.t. x(t) + b > 0)
θ = 1.0                 # threshold (when reached, we spike)
κ = (b+offset)/rate/θ   # capacitance (integrator weight)
Δ = 0.0                 # TODO refractory?
l = 0.0                 # TODO leak?

#=== decoder parameters ====#
n = 10                  # n filters in total (equal for both models)
η = 1e-4                # learning rate      (equal for both models)
α = 2                   # fastest decay rate
λ = 2                   # fastest decay rate
λs = range(1, length=n, stop=0.01)

ramp(z) = log(1 + ℯ^z)  # non-linearity (ramp)
sigmoid(z) = ℯ^z / (1 + ℯ^z) # derivative of ramp
sig_der(z) = ℯ^(-z) / (1 + ℯ^(-z))^2 # derivative of sigmoid

#=== simulation ============#
training_cycles = 10000  # how many total periods to see in training
test_cycles = 3          # and how many in the test phase

function init_task(task)
    """
    Returns an io tuple (of input x and target z) and the target period.
    The target period is just the lowest period (lp) for a sine-based task
    and the least common multiple of the periods in a mix task.
    """
    Random.Random.seed!(1) # rng seed, for model comparability

    io = () # a tuple of functions x (input) and z (target)

    if task == "sin"       # reconstruct sine
        let input
            input(t) = sin(2π/lp*t) + offset
            io = (x=input, z=t->input(t+latency))
        end
        tp = lp
    elseif task == "mix2"  # reconstruct a mix of 2 sines
        let input
            input(t) = sin(2π/hp*t) + sin(2π/lp*t) + offset
            io = (x=input,  z=t->input(t+latency))
        end
        tp = lcm([lp, hp]) # total period i
    elseif task == "mix3"  # reconstruct a mix of 3 sines
        let input
            input(t) = sin(2π/hp*t) + sin(2π/mp*t) + sin(2π/lp*t) + offset
            io = (x=input,  z=t->input(t+latency))
        end
        tp = lcm([lp, mp, hp]) # total period is lcm of periods
    elseif task == "1of3"  # demixing: extract 1 component from a mix3
        let input, target
            input(t) = sin(2π/hp*t) + sin(2π/mp*t) + sin(2π/lp*t) + offset
            target(t) = sin(2π/mp*t) + offset
            io = (x=input, z=t->target(t+latency))
        end
        tp = mp
    elseif task == "shift" # shift sine by phase
        let input
            input(t) = sin(2π/lp*t) + offset
            io = (x=input, z=t->input(t+latency+phase))
        end
        tp = lp
    elseif task == "noise" # denoise: reconstruct sine with white noise
        let input
            input(t) = sin(2π/lp*t) + offset
            noise = randn(1000) .* noise_std # noise with µ=0
            io = (x=t->input(t) + noise[Int(floor(t*100))%1000+1],
                  z=t->input(t+latency))
        end
        tp = lp
    elseif task == "abs"   # take absolute value
        let input
            input(t) = sin(2π/lp*t) + offset
            io = (x=input, z=t->abs(input(t+latency)-offset)+offset)
        end
        tp = lp
    elseif task == "relu"  # take maximum of 0 and signal
        let input
            input(t) = sin(2π/lp*t) + offset
            io = (x=input, z=t->Base.max(0., input(t+latency)-offset)+offset)
        end
        tp = lp
    elseif task == "sqr"   # square of input
        let input
            input(t) = sin(2π/lp*t) + offset
            io = (x=input, z=t->(input(t+latency)-offset)^2+offset)
        end
        tp = lp
    elseif task == "sign"  # binary transformation: sign of input
        let input
            input(t) = sin(2π/lp*t) + offset
            io = (x=input, z=t->sign(input(t+latency)-offset)+offset)
        end
        tp = lp
    end

    global tp = tp
    return io, tp
end

function init_times(cycles, tp)
    """
    `Cycles` is how many total periods (`tp`) of the signal are presented.
    According to the `show` variable, the saving times (`tsave`) are set.
    Returns the list of save points `tsave`.
    """
    tmax = tp * cycles

    if show in ["w_evol", "mse_evol"]
        tsave = 0.:tp:tmax    # save regularly (once / period)
    else
        tsave = [0.,tmax]     # save only first and last state
    end

    return tsave
end

function init_state(; range=.05)
    """
    Initializes taps and weights randomly from a normal disrtibution
    centered around zero with standard deviation of `range`.
    Note that there are n+1 weights, the last one is the bias weight.
    Returns the initial state being in the form [ys; ws; 0].
    """
    Random.Random.seed!(1)         # rng seed, for comparability
    w0 = (.5 .- rand(n+1))*2*range # random initial weights
    y0 = (.5 .- rand(n))*2*range   # initial taps
    u0 = [y0; w0; 0.0]             # initial state: ys, ws, X
    return u0
end

function init_training(task, cycles)
    """
    Wrapper function around `init_task`, `init_times` and `init_state`.
    """
    # some global settings
    global mode = "train"
    global λs = range(λ, length=n, stop=0.01) # for the lazar decoder

    io, tp = init_task(task)
    u0 = init_state()
    tsave = init_times(cycles, tp)
    return io, tp, u0, tsave
end

#=== ODE DEFINITIONS =======#

function af(du, u, io, t)
    """
    System of ODE to model communication between two AF neurons,
    one reduced on the encoding, the other reduced on the decoding part.

    input signal → enc → channel (spikes) → dec → reconstructed signal

    The state has the following form: (y_1, …, y_n, w_1, …, w_n, w_b, X).
    Note that the spiking part (X) belongs to the presynaptic
    and the filter bank (y_1, …, w_n, w_b) to the postsynaptic neuron.
    """

    # decoder (postsynaptic)
    banklen = n ÷ 2                  # two filter banks of the same size

    # filter bank 1
    du[1] = -α * u[1]                # no input here (generated by events)
    du[2:banklen] .= -α * u[2:banklen] .+ α * u[1:banklen-1]

    # decoder output ("potential")
    z = u[1:n]' * u[n+1:2n] + u[2n+1]

    # filter bank 2
    du[banklen+1] = -α * u[banklen+1] + α * z # input from previous output z
    du[banklen+2:n] .= -α * u[banklen+2:n] .+ α * u[banklen+1:n-1]

    # weight updates
    if mode == "train"
        du[n+1:2n] = 2 * η * (io.z(t) - z) .* u[1:n]
        du[2n+1]   = 2 * η * (io.z(t) - z) # bias weight
    end

    # encoder (presynaptic)
    du[end] = (io.x(t) + b) / κ    # rectangular integration of signal
end

function naf(du, u, io, t)
    """
    Nonlinear AF, same as AF but with ramp function.

    System of ODE to model communication between two AF neurons,
    one reduced on the encoding, the other reduced on the decoding part.

    input signal → enc → channel (spikes) → dec → reconstructed signal

    The state has the following form: (y_1, …, y_n, w_1, …, w_n, w_b, X).
    Note that the spiking part (X) belongs to the presynaptic
    and the filter bank (y_1, …, w_n, w_b) to the postsynaptic neuron.
    """

    # decoder (postsynaptic)
    banklen = n ÷ 2                  # two filter banks of the same size

    # filter bank 1
    du[1] = -α * u[1]                # no input here (generated by events)
    du[2:banklen] .= -α * u[2:banklen] .+ α * u[1:banklen-1]

    # decoder output ("potential")
    z = u[1:n]' * u[n+1:2n] + u[2n+1]

    # filter bank 2
    du[banklen+1] = -α * u[banklen+1] + α*z # input from previous output z
    du[banklen+2:n] .= -α * u[banklen+2:n] .+ α * u[banklen+1:n-1]

    # weight updates
    if mode == "train"
        w_b = u[end-1] # bias weight for shift of ramp
        du[n+1:2n] = (2 * η * (io.z(t) - ramp(z+w_b)) .* u[1:n]
                      * sigmoid(z+w_b))
        # update for w_b (multiplied with n for stronger learning)
        du[end-1]  = 2 * n * η * (io.z(t) - ramp(z+w_b)) * sigmoid(z+w_b)
    end

    # encoder (presynaptic)
    du[end] = (io.x(t) + b) / κ    # rectangular integration of signal
end

function lazar(du, u, io, t)
    """
    Implementation of [Lazar 2006: A simple model for spike processing].
    A zeroth-order approximation of the stimulus is computed
    by spike weighting and filtering the signal.

    A decoding filter bank is constructed from n parallel filters
    each of which gets the same input from the presynaptic neuron.

    Filters have different time constants λ_i stored in λs.
    The output is the weighted sum of the filters, ∑_i w_i · y_i.

    The state has the following form: (y_1, …, y_n, w_1, …, w_n, w_b, X).
    Note that the spiking part (X) belongs to the presynaptic
    and the filter bank (y_1, …, w_n, w_b) to the postsynaptic neuron.
    """

    # decoder (postsynaptic)
    du[1:n] .= -λs .* u[1:n]

    # output
    z = u[1:n]' * u[n+1:2n] + u[2n+1]

    # weight updates
    if mode == "train"
        du[n+1:2n] .= 2 * η * (io.z(t) - z) .* u[1:n]
        du[2n+1]    = 2 * η * (io.z(t) - z) # bias weight
    end

    # encoder (presynaptic)
    du[end] = (io.x(t) + b) / κ # rectangular integration of signal
end

#=== ACCESSOR UTILS ==========#

function z(sol, t)
    u = sol(t)
    y = u[1:n]
    w = u[n+1:2n]
    w_b = u[2n+1] # bias weight
    w' * y + w_b
end

function r(sol, t)
    u = sol(t)
    ramp(z(sol, t))
end

function out(sol, t)
    if algo == :naf
        r(sol, t)
    else
        z(sol, t)
    end
end

function y_i(sol, i)
    function y(t)
        sol(t)[i]
    end
end

function w_i(sol, i)
    function w(t)
        sol(t)[n+i]
    end
end

function mse(sol, io, period)
    """
    Calculate mean squared error (normalized by signal amplitude)
    of the solution output w.r.t. the target function
    over the given period (array of time points)
    and normalize it by the std of the target amplitude.
    """
    mse = sum([(out(sol,t) - io.z(t))^2 for t in period]) /length(period)

    # compute std of target (on 1000 points over tp) to normalize mse
    var = StatsBase.var([io.z(t) for t in period])
    error = mse / var
    round(error, digits=5) # round on 5 decimal digits
end

#=== SOLVER FUNCTIONS ======#

function run(io, u0, tsave, save_spikes=false)
    """
    Runs the model `algo` in `mode` (train or test), i.e.
    it solves the specified ODE over the time specified in `tsave`.
    Returns the solution of the solver and if appropriate
    (`save_spikes == true`), also a list of all spike times.
    """
    tspan = (tsave[1], tsave[end])       # run over the whole save period
    tspike = undef                       # time of last spike, no initial
    spikes = []                          # list to save all spike times
    function threshold(u, t, integrator) # callback condition
        return u[end] - θ
    end
    function spike!(integrator)          # callback handler
        if algo == :lazar
            if tspike != undef           # ignore first spike (weight issue)
                isi = integrator.t - tspike  # inter-spike interval
                c_k = κ * θ - b * isi        # weighting for spikes
                integrator.u[1:n] .+= c_k    # add spike upon each tap
            end
            tspike = integrator.t        # save time of spike
        else                             # for AF/nAF, just do:
            integrator.u[1]  += 1.0      # input to postsynaptic neuron
        end
        integrator.u[end] = 0.0          # reset X integrator
        if save_spikes
            push!(spikes, integrator.t)  # save spike time
        end
    end
    fire = ContinuousCallback(threshold, spike!, # not to save u at spikes:
                              save_positions=(false, false))
    prob = ODEProblem(eval(algo), u0, tspan, io, callback=fire)
    sol = solve(prob, saveat=tsave, maxiters=Inf)
    if save_spikes
        return (sol, spikes) # solution and all spikes
    end
    return sol
end

function test_error(sol, io, tp)
    """
    Tests the last state of a solution `sol`,
    saves for `100/tp * test_cycles` times and
    returns the new solution object and the error.
    """
    global mode = "test"
    u_end, t_end = sol[end], sol.t[end]
    tsave2 = t_end:.01:t_end+test_cycles*tp
    sol2   = run(io, u_end, tsave2)
    return sol2, mse(sol2, io, tsave2)
end

function impulse_response(sol; dur=10)
    """
    Compute impulse response of a trained model (`sol`).
    The model is fully specified by its weight vector.
    Returns a solution for the i.r. over the duration `dur`.
    """
    w_end = sol[end][n+1:2n+1]
    u = zeros(2n+2)       # make new fresh state
    u[n+1:2n+1] = w_end   # take learned model
    if algo == :lazar
        u[1:n] .= 1       # put initial spike into all taps
    else
        u[1] = 1          # initial spike y_1 = 1
    end
    global mode = "test"  # do not adapt weights in rerun
    io = (x=t->0, )       # use no input and no z, because mode == "test"
    # Note that no encoder is necessary, therefore no callback.
    prob = ODEProblem(eval(algo), u, (0.,dur), io)
    solve(prob, saveat=0:.01:dur, maxiters=Inf)
end

function running_mse(sol, io, save_periods=[])
    """
    Takes a solution object and tests the model for each stored state.
    `save_periods` is a list of periods for which the solution is saved.
    Returns a list of MSEs and a list of the saved solutions, if any.
    """
    mses   = []
    sols   = []
    period = 0
    global mode = "test"
    for (u,t) in zip(sol.u, sol.t)
        tsave2 = t:tp/100:t+test_cycles*tp
        sol2 = run(io, u, tsave2)
        push!(mses, mse(sol2, io, tsave2))
        if period in save_periods
            push!(sols, sol2)
        end
        period += 1
    end
    return mses, sols
end

#=== BASIC PLOTS =============#

function palgo()
    """ Prettifies the algo name for plots. """
    if algo == :af
        "AF"
    elseif algo == :lazar
        "Lazar"
    elseif algo == :naf
        "nAF"
    end
end

function evol_plot(show, sol, io)
    """ Returns a plot. """
    if !(show in ["w_evol", "y_evol", "mse_evol"]) return end
    if show == "w_evol"
        plot([w_i(sol, i) for i in 1:n+1], sol.t,
             labels=["w_$i" for i in 1:n+1],
             xlabel="time",
             title="$(palgo()), weights", titleloc=:left)
    elseif show == "y_evol"
        plot([w_i(sol, i) for i in 1:n+1], sol.t,
             labels=["y_$i" for i in 1:n+1],
             xlabel="time",
             title="$(palgo()), taps")
    elseif show == "mse_evol"
        mses, _ = running_mse(sol, io)
        plot(mses,
             leg=false, title="$algo (final MSE: $(mses[end]))",
             ylabel="mse", xlabel="periods")
    end
end

function end_plot(show, sol, io, tp; return_mse=false)
    """ Returns a plot and, if `return_mse==true` also the error. """
    if !(show in ["out", "zyw", "y*w", "y"]) return end

    sol2, error = test_error(sol, io, tp)

    plt = nothing
    if show == "out"
        plt = plot([t->out(sol2, t), io.z], sol2.t,
                   labels=["approx", "target"],
                   title="$algo (MSE: $error)")
    elseif show == "y"
        plt = plot([y_i(sol2, i) for i in 1:n], sol2.t,
                   labels=["y_$i" for i in 1:n], leg=:bottomright,
                   xticks=sol2.t[1]:tp:sol2.t[end], xlabel="time",
                   title="$(palgo()), taps", titleloc=:left)
    elseif show == "y*w"
        plt = plot([t->(y_i(sol2, i)(t) * w_i(sol2, i)(t)) for i in 1:n],
                   sol2.t,
                   labels=["y_$i" for i in 1:n], leg=:bottomright,
                   xticks=sol2.t[1]:tp:sol2.t[end], xlabel="time",
                   title="$(palgo()), taps")
    elseif show == "zyw"
        plt = plot([t->z(sol2, t), y_i(sol2, 1), w_i(sol2, 1),
                    io.z,          y_i(sol2, n), w_i(sol2, n)],
                   sol2.t,
                   label=["approx" "y_1" "w_1" "target" "y_n" "w_n"],
                   title=["$algo (MSE: $error)" "taps" "weights"],
                   layout=(3,1))
    end

    if return_mse
        return plot, error
    end
    return plt
end

function imp_plot(show, sol)
    """ Returns a plot. """
    if !(show in ["imp", "impi"]) return end
    sol_imp = impulse_response(sol)

    if show == "imp"
        plot(t->z(sol_imp, t),
             sol_imp.t,
             label="total imp", xlabel="t",
             title="$algo, impulse response")
    elseif show == "impi"
        plot([y_i(sol_imp, i) for i in 1:n],
             sol_imp.t,
             labels=["y$i" for i in 1:n], xlabel="t",
             title="$algo, individual impulse response")
    end
end

#=== RUNNING + SAVING ======#

function serialize(u_end, filename, error)
    dict = Dict("algo"    => algo,
                "task"    => task,
                "n"       => n,
                "eta"     => η,
                "alpha"   => α,
                "latency" => latency,
                "mse"     => error,
                "u"       => u_end
                )
    string = JSON.json(dict)
    open(filename, "w") do f
        write(f, string)
    end
end

if exec
    # init
    io, tp, u0, tsave = init_training(task, training_cycles)

    # train
    sol = run(io, u0, tsave)

    # plot
    evol_plot(show, sol, io)
    _, error = end_plot(show, sol, io, tp, return_mse=true)
    imp_plot(show, sol)
    println("MSE: ", error)

    # save figure
    fn = "$algo-$task-$show-$mark-alpha$α-latency$latency-eta$η-offset$offset-t$(tsave[end])"
    dir = task
    savefig("../figures/$dir/$fn.png")

    # serialize the model
    #if mode == "train"
    #    serialize(sol1, "../models/$fn", error)
    #end
end
