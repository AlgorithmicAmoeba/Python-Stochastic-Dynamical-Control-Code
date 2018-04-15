# Linear plant controlled with a linear MPC using a KF to estimate the state.
# Conventional deterministic constraints.

import closedloop_scenarios_single.closedloop_params
import src.LQR as LQR
import src.LLDS as LLDS
import scipy.stats
import src.MPC as MPC
import src.Results as Results
import numpy
import matplotlib.pyplot as plt
import src.PF as PF


def main(mcN=1, linear=True, pf=False):
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
    x_off, usp = LQR.offset(A, numpy.matrix(B), params.C2, H, numpy.matrix([ysp]))  # control offset
    ysp = x_off
    usp = numpy.array([usp])

    # Noise distributions
    state_noise_dist = scipy.stats.multivariate_normal(cov=params.Q)
    meas_noise_dist = scipy.stats.multivariate_normal(cov=params.R2)

    # PF functions
    def f(x, u, w):
        return params.cstr_model.run_reactor(x, u, params.h) + w

    def g(x):
        return params.C2 @ x  # state observation

    cstr_pf = PF.Model(f, g)

    # Set up the KF
    kf_cstr = LLDS.LLDS(A, B, params.C2, params.Q, params.R2)  # set up the KF object (measuring both states)

    # Setup MPC
    horizon = 150

    # add state constraints
    aline = 10  # slope of constraint line ax + by + c = 0
    if linear:
        cline = -411  # negative of the y axis intercept
    else:
        cline = -404  # negative of the y axis intercept
    bline = 1

    if linear:
        lim_u = 10000
    else:
        lim_u = 20000

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
            temp = kf_cstr.init_filter(init_state - b, params.init_state_covar, params.ys2[:, 0] - b)  # filter
            params.kfmeans[:, 0], params.kfcovars[:, :, 0] = temp

        if pf:
            nP = 200
            prior_dist = scipy.stats.multivariate_normal(mean=init_state,
                                                         cov=params.init_state_covar)  # prior distribution

            particles = PF.init_pf(prior_dist, nP, 2)  # initialise the particles

            particles = PF.init_filter(particles, params.ys2[:, 0], meas_noise_dist, cstr_pf)
            params.pfmeans[:, 0], params.pfcovars[:, :, 0] = PF.get_stats(particles)

            params.us[0] = MPC.mpc_mean(params.pfmeans[:, 0]-b, horizon, A, numpy.matrix(B), b,
                                        aline, bline, cline, params.QQ, params.RR, ysp,
                                        usp[0], lim_u, 1000.0)  # get the controller input
        else:
            params.us[0] = MPC.mpc_mean(params.kfmeans[:, 0], horizon, A, numpy.matrix(B), b,
                                        aline, bline, cline, params.QQ, params.RR, ysp,
                                        usp[0], lim_u, 1000.0)  # get the controller input
        for t in range(1, params.N):
            if linear:
                params.xs[:, t] = A @ params.xs[:, t - 1] + B * params.us[t - 1] + state_noise_dist.rvs()
                params.ys2[:, t] = params.C2 @ params.xs[:, t] + meas_noise_dist.rvs()  # measure from actual plant
                params.kfmeans[:, t], params.kfcovars[:, :, t] = kf_cstr.step_filter(params.kfmeans[:, t-1],
                                                                                     params.kfcovars[:, :, t-1],
                                                                                     params.us[t-1], params.ys2[:, t])
            else:
                params.xs[:, t] = params.cstr_model.run_reactor(params.xs[:, t - 1], params.us[t - 1], params.h)
                params.xs[:, t] += state_noise_dist.rvs()  # actual plant
                params.ys2[:, t] = params.C2 @ params.xs[:, t] + meas_noise_dist.rvs()  # measure from actual plant
                params.kfmeans[:, t], params.kfcovars[:, :, t] = kf_cstr.step_filter(params.kfmeans[:, t-1],
                                                                                     params.kfcovars[:, :, t-1],
                                                                                     params.us[t-1],
                                                                                     params.ys2[:, t]-b)
            if pf:
                PF.pf_filter(particles, params.us[t - 1], params.ys2[:, t], state_noise_dist, meas_noise_dist, cstr_pf)
                params.pfmeans[:, t], params.pfcovars[:, :, t] = PF.get_stats(particles)

            if t % 10 == 0:
                if pf:
                    params.us[t] = MPC.mpc_mean(params.pfmeans[:, t]-b, horizon, A, numpy.matrix(B), b,
                                                aline, bline, cline, params.QQ,
                                                params.RR, ysp, usp[0], lim_u, 1000.0)
                else:
                    params.us[t] = MPC.mpc_mean(params.kfmeans[:, t], horizon, A, numpy.matrix(B), b,
                                                aline, bline, cline, params.QQ,
                                                params.RR, ysp, usp[0], lim_u, 1000.0)
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
            if pf:
                Results.plot_tracking1(params.ts, params.xs, params.ys2, params.pfmeans, params.us, 2, ysp[0] + b[0])
                Results.plot_ellipses2(params.ts, params.xs, params.pfmeans, params.pfcovars,
                                       [aline, cline], linsystems[1].op, True, -2.0 * numpy.log(1 - 0.9), 1, "best")
            else:
                Results.plot_tracking1(params.ts, params.xs, params.ys2, params.kfmeans, params.us, 2, ysp[0]+b[0])
                Results.plot_ellipses2(params.ts, params.xs, params.kfmeans, params.kfcovars,
                                       [aline, cline], linsystems[1].op, True, -2.0*numpy.log(1-0.9), 1, "best")
            Results.check_constraint(params.ts, params.xs, [aline, cline])
            Results.calc_error1(params.xs, ysp[0]+b[0])
            Results.calc_energy(params.us, 0.0)
            plt.show()

    if mcN != 1:
        print("The absolute MC average error is: ", sum(abs(mcerrs)) / mcN)
        if linear:
            if pf:
                numpy.savetxt("linmod_pf_mean_mc2.csv", xconcen, delimiter=",")
            else:
                numpy.savetxt("linmod_kf_mean_mc2.csv", xconcen, delimiter=",")
        else:
            if pf:
                numpy.savetxt("nonlinmod_pf_mean_mc2.csv", xconcen, delimiter=",")
            else:
                numpy.savetxt("nonlinmod_kf_mean_mc2.csv", xconcen, delimiter=",")

    nocount = len(mcdists[0]) - numpy.count_nonzero(mcdists[0])
    filteredResults = numpy.zeros([2, mcN - nocount])
    counter = 0
    for k in range(mcN):
        if mcdists[0, k] != 0.0:
            filteredResults[:, counter] = mcdists[:, k]
            counter += 1

    if linear:
        if pf:
            numpy.savetxt("linmod_pf_mean_mc.csv", filteredResults, delimiter=",",
                          fmt="%f")
        else:
            numpy.savetxt("linmod_kf_mean_mc.csv", filteredResults, delimiter=",",
                          fmt="%f")
    else:
        if pf:
            numpy.savetxt("nonlinmod_pf_mean_mc.csv", filteredResults, delimiter=",",
                          fmt="%f")
        else:
            numpy.savetxt("nonlinmod_kf_mean_mc.csv", filteredResults, delimiter=",",
                          fmt="%f")
