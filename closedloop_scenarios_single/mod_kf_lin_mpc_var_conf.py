# Linear Plant controlled with a linear MPC using a KF to estimate the state.
# Stochastic constraints.

import closedloop_scenarios_single.closedloop_params
import src.LQR as LQR
import src.LLDS as LLDS
import scipy.stats
import src.MPC as MPC
import src.Results as Results
import numpy
import matplotlib.pyplot as plt


def main(nine, mcN=1, linear=True):
    if nine == 90:
        k_squared = 4.6052
        plot_setting = 0
    elif nine == 99:
        k_squared = 9.21
        plot_setting = 1
    elif nine == 999:
        k_squared = 13.8155
        plot_setting = 2
    else:
        return None

    tend = 80
    params = closedloop_scenarios_single.closedloop_params.Params(tend)  # end time of simulation

    # Get the linear model
    linsystems = params.cstr_model.get_nominal_linear_systems(params.h)  # cstr_model comes from params.jl
    opoint = 1  # the specific operating point we are going to use for control

    init_state = numpy.array([0.55, 450])  # random initial point near operating point

    # Set the state space model
    A = linsystems[opoint].A
    B = linsystems[opoint].B
    b = linsystems[opoint].b  # offset from the origin

    # Set point
    ysp = linsystems[opoint].op[0] - b[0]  # Medium concentration
    H = numpy.matrix([1, 0])  # only attempt to control the concentration
    x_off, usp = LQR.offset(A, numpy.matrix(B), params.C2, H, numpy.array([ysp]))  # control offset
    ysp = x_off
    usp = numpy.array([usp])

    # Set up the KF
    kf_cstr = LLDS.LLDS(A, B, params.C2, params.Q, params.R2)  # set up the KF object (measuring both states)
    state_noise_dist = scipy.stats.multivariate_normal(cov=params.Q)
    meas_noise_dist = scipy.stats.multivariate_normal(cov=params.R2)

    # Setup MPC
    horizon = 150
    # add state constraints
    aline = 10  # slope of constraint line ax + by + c = 0
    cline = -411  # negative of the y axis intercept
    bline = 1
    e = cline

    growvar = True
    if linear:
        limu = 10000
    else:
        limu = 20000

    mcdists = numpy.zeros([2, mcN])
    xconcen = numpy.zeros([params.N, mcN])
    mcerrs = numpy.zeros(mcN)

    for mciter in range(mcN):
        # First time step of the simulation
        if linear:
            params.xs[:, 0] = init_state - b  # set simulation starting point to the random initial state
            params.ys2[:, 0] = params.C2 @ params.xs[:, 0] + meas_noise_dist.rvs()  # measure from actual plant
            temp = kf_cstr.init_filter(init_state - b, params.init_state_covar, params.ys2[:, 0])  # filter
            params.kfmeans[:, 0], params.kfcovars[:, :, 0] = temp
        else:
            params.xs[:, 0] = init_state  # set simulation starting point to the random initial state
            params.ys2[:, 0] = params.C2 @ params.xs[:, 0] + meas_noise_dist.rvs()  # measure from actual plant
            temp = kf_cstr.init_filter(init_state - b, params.init_state_covar, params.ys2[:, 0]-b)  # filter
            params.kfmeans[:, 0], params.kfcovars[:, :, 0] = temp

        params.us[0] = MPC.mpc_var(params.kfmeans[:, 0], params.kfcovars[:, :, 0], horizon,
                                   A, numpy.matrix(B), b, aline, bline, e, params.QQ, params.RR, ysp,
                                   usp[0], limu, 1000.0, params.Q, k_squared, growvar)  # get the controller input
        for t in range(1, params.N):
            if linear:
                params.xs[:, t] = A @ params.xs[:, t-1] + B*params.us[t-1] + state_noise_dist.rvs()  # actual plant
                params.ys2[:, t] = params.C2 @ params.xs[:, t] + meas_noise_dist.rvs()  # measure from actual plant
                params.kfmeans[:, t], params.kfcovars[:, :, t] = kf_cstr.step_filter(params.kfmeans[:, t - 1],
                                                                                     params.kfcovars[:, :, t - 1],
                                                                                     params.us[t - 1], params.ys2[:, t])
            else:
                params.xs[:, t] = params.cstr_model.run_reactor(params.xs[:, t - 1], params.us[t - 1], params.h, )
                params.xs[:, t] += state_noise_dist.rvs()  # actual plant
                params.ys2[:, t] = params.C2 @ params.xs[:, t] + meas_noise_dist.rvs()  # measure from actual plant

                params.kfmeans[:, t], params.kfcovars[:, :, t] = kf_cstr.step_filter(params.kfmeans[:, t-1],
                                                                                     params.kfcovars[:, :, t-1],
                                                                                     params.us[t-1], params.ys2[:, t]-b)
            if t % 10 == 0:
                params.us[t] = MPC.mpc_var(params.kfmeans[:, t], params.kfcovars[:, :, t], horizon,
                                           A, numpy.matrix(B), b, aline, bline, e, params.QQ, params.RR, ysp,
                                           usp[0], limu, 1000, params.Q, k_squared, growvar)  # get controller input

                if params.us[t] is None or numpy.isnan(params.us[t]):
                    break
            else:
                params.us[t] = params.us[t-1]

        for i in range(len(params.kfmeans[0])):
            params.kfmeans[:, i] += b
            if linear:
                params.xs[:, i] += b
                params.ys2[:, i] += b

        if mcN > 1:
            xconcen[:, mciter] = params.xs[0, :]
            mcerrs[mciter] = Results.calc_error1(params.xs, ysp[0] + b[0])
            Results.get_mc_res(params.xs, params.kfcovars, [aline, cline], mcdists, mciter, params.h)
        if mcN == 1:
            # Plot the results
            Results.plot_tracking1(params.ts, params.xs, params.ys2, params.kfmeans, params.us, 2, ysp[0]+b[0])
            Results.plot_ellipses2(params.ts, params.xs, params.kfmeans, params.kfcovars, [aline, cline],
                                   linsystems[opoint].op, True, k_squared, plot_setting, "best")
            Results.check_constraint(params.ts, params.xs, [aline, cline])
            Results.calc_error1(params.xs, ysp[0]+b[0])
            Results.calc_energy(params.us, 0.0)
            plt.show()

    if mcN != 1:
        print("The absolute MC average error is: ", sum(abs(mcerrs)) / mcN)
        if linear:
            numpy.savetxt("linmod_kf_var{}_mc.csv".format(nine), xconcen, delimiter=",")
        else:
            numpy.savetxt("nonlinmod_kf_var{}_mc.csv".format(nine), xconcen, delimiter=",")
