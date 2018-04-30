# Control using two nonlinear models and measuring both states

import numpy
import scipy.stats
import closedloop_scenarios_single.closedloop_params as params
import src.Results as Results
import src.LQR as LQR
import src.MPC as MPC
import src.SPF as SPF
import src.RBPF as RBPF
import typing

tend = 200
params = params.Params(tend)

mcN = 50
mcdists = numpy.zeros([2, mcN])
xconcen = numpy.zeros([params.N, mcN])
mcerr = numpy.zeros(mcN)
mciter = -1
while mciter < mcN-1:
    isDone = True
    init_state = numpy.array([0.55, 450])  # initial state

    # Setup Switching Particle Filter
    A = numpy.array([[0.999, 0.001],
                     [0.001, 0.999]])


    def fun1(x, u, w):
        return params.cstr_model.run_reactor(x, u, params.h) + w


    def fun2(x, u, w):
        return params.cstr_model_broken.run_reactor(x, u, params.h) + w


    def gs(x):
        return params.C2 @ x


    F = [fun1, fun2]
    G = [gs, gs]
    numSwitches = 2

    ydists = numpy.array(
        [scipy.stats.multivariate_normal(cov=params.R2), scipy.stats.multivariate_normal(cov=params.R2)])
    xdists = numpy.array([scipy.stats.multivariate_normal(cov=params.Q), scipy.stats.multivariate_normal(cov=params.Q)])
    cstr_filter = SPF.Model(F, G, A, xdists, ydists)

    nP = 500  # number of particles
    xdist = scipy.stats.multivariate_normal(mean=init_state, cov=params.init_state_covar)
    sdist = [0.9, 0.1]
    particles = SPF.init_spf(xdist, sdist, nP, 2)

    switchtrack = numpy.zeros([2, params.N])
    maxtrack = numpy.zeros([numSwitches, params.N])
    smoothedtrack = numpy.zeros([numSwitches, params.N])

    state_noise_dist = scipy.stats.multivariate_normal(cov=params.Q)
    meas_noise_dist = scipy.stats.multivariate_normal(cov=params.R2)

    # Setup control (use linear control)
    linsystems = params.cstr_model.get_nominal_linear_systems(params.h)
    linsystems_broken = params.cstr_model_broken.get_nominal_linear_systems(params.h)
    opoint = 1  # the specific linear model we will use

    lin_models = [None] * 2  # type: typing.List[RBPF.Model]
    lin_models[0] = RBPF.Model(linsystems[opoint].A, linsystems[opoint].B, linsystems[opoint].b,
                               params.C2, params.Q, params.R2)
    lin_models[1] = RBPF.Model(linsystems_broken[opoint].A, linsystems_broken[opoint].B, linsystems_broken[opoint].b,
                               params.C2, params.Q, params.R2)

    H = numpy.matrix([1, 0])  # only attempt to control the concentration
    setpoint = 0.49
    controllers = [None] * 2  # type: typing.List[LQR.Controller]
    for k in range(2):
        ysp = setpoint - lin_models[k].b[0]  # set point is set here
        x_off, u_off = LQR.offset(lin_models[k].A, numpy.matrix(lin_models[k].B), params.C2, H, numpy.matrix([ysp]))
        K = LQR.lqr(lin_models[k].A, numpy.matrix(lin_models[k].B), params.QQ, params.RR)
        controllers[k] = LQR.Controller(K, x_off, u_off)

    # Setup simulation
    params.xs[:, 0] = init_state
    params.ys2[:, 0] = params.C2 @ params.xs[:, 0] + meas_noise_dist.rvs()  # measured from actual plant

    SPF.init_filter(particles, params.ys2[:, 0], cstr_filter)

    for k in range(numSwitches):
        switchtrack[k, 0] = numpy.sum(particles.w[numpy.where(particles.s == k)[0]])

    maxtrack[:, 0] = SPF.get_max_track(particles, numSwitches)
    smoothedtrack[:, 0] = RBPF.smoothed_track(numSwitches, switchtrack, 1, 10)

    params.spfmeans[:, 0], params.spfcovars[:, :, 0] = SPF.get_stats(particles)

    # Controller Input
    ind = numpy.argmax(maxtrack[:, 0])  # use this model and controller
    horizon = 150
    params.us[0] = MPC.mpc_lqr(params.spfmeans[:, 0] - lin_models[ind].b, horizon, lin_models[ind].A,
                               numpy.matrix(lin_models[ind].B), params.QQ, params.RR,
                               controllers[ind].x_off, controllers[ind].u_off)

    # Loop through the rest of time
    for t in range(1, params.N):
        random_element = state_noise_dist.rvs()
        if params.ts[t] < 100:  # break here
            params.xs[:, t] = params.cstr_model.run_reactor(params.xs[:, t - 1], params.us[t - 1],
                                                            params.h) + random_element
        else:
            params.xs[:, t] = params.cstr_model_broken.run_reactor(params.xs[:, t - 1], params.us[t - 1], params.h)
            params.xs[:, t] += random_element

        params.ys2[:, t] = params.C2 @ params.xs[:, t] + meas_noise_dist.rvs()  # measured from actual plant

        SPF.spf_filter(particles, params.us[t - 1], params.ys2[:, t], cstr_filter)
        params.spfmeans[:, t], params.spfcovars[:, :, t] = SPF.get_stats(particles)

        for k in range(numSwitches):
            switchtrack[k, 0] = numpy.sum(particles.w[numpy.where(particles.s == k)[0]])

        maxtrack[:, t] = SPF.get_max_track(particles, numSwitches)
        smoothedtrack[:, t] = RBPF.smoothed_track(numSwitches, switchtrack, t, 40)

        # Controller Input
        if t % 1 == 0:
            ind = numpy.argmax(maxtrack[:, t])  # use this model and controller
            params.us[t] = MPC.mpc_lqr(params.spfmeans[:, t] - lin_models[ind].b, horizon, lin_models[ind].A,
                                       numpy.matrix(lin_models[ind].B), params.QQ, params.RR,
                                       controllers[ind].x_off, controllers[ind].u_off)
        else:
            params.us[t] = params.us[t - 1]
            if params.us[t] is None or numpy.isnan(params.us[t]):
                isDone = False
                break

        if isDone:
            mciter += 1
            mcerr[mciter] = Results.calc_error1(params.xs, setpoint)
            Results.get_mc_res(params.xs, params.spfcovars, [10, -410], mcdists, mciter, params.h)
            xconcen[:, mciter] = params.xs[0, :]

    mcave = sum(abs(mcerr)) / mcN
    print("Monte Carlo average concentration error: ", mcave)

nocount = len(mcdists[0]) - numpy.count_nonzero(mcdists[0])
filteredResults = numpy.zeros([2, mcN - nocount])
counter = 0
for k in range(mcN):
    if mcdists[0, k] != 0.0:
        filteredResults[:, counter] = mcdists[:, k]
        counter += 1

numpy.savetxt("spf_lqg.csv.csv", filteredResults, delimiter=",", fmt="%f")
numpy.savetxt("spf_lqg_mc2.csv.csv", xconcen, delimiter=",", fmt="%f")
