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
import src.PF as PF
import src.Auxiliary as Auxiliary


def main(nine, mcN=1, linear=True, pf=False, numerical=False):
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
    x_off, usp = LQR.offset(A, numpy.matrix(B), params.C2, H, numpy.matrix([ysp]))  # control offset
    ysp = x_off
    usp = numpy.array([usp])

    # PF functions
    def f(x, u, w):
        return params.cstr_model.run_reactor(x, u, params.h) + w

    def g(x):
        return params.C2 @ x  # state observation

    cstr_pf = PF.Model(f, g)

    nP = 200
    if numerical:
        nP = 500

    # Set up for numerical
    Ndiv = len(range(0, tend, 3))
    kldiv = numpy.zeros(Ndiv)  # Kullback-Leibler Divergence as a function of time
    basediv = numpy.zeros(Ndiv)  # Baseline
    unidiv = numpy.zeros(Ndiv)  # Uniform comparison
    klts = numpy.zeros(Ndiv)
    ndivcounter = 0
    temp_states = numpy.zeros([2, nP])

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

    mciter = -1

    particles = None
    while mciter < mcN-1:
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

        if pf:
            if linear:
                prior_dist = scipy.stats.multivariate_normal(mean=init_state-b,
                                                             cov=params.init_state_covar)  # prior distribution
            else:
                prior_dist = scipy.stats.multivariate_normal(mean=init_state,
                                                             cov=params.init_state_covar)  # prior distribution
            particles = PF.init_pf(prior_dist, nP, 2)  # initialise the particles

            particles = PF.init_filter(particles, params.ys2[:, 0], meas_noise_dist, cstr_pf)
            params.pfmeans[:, 0], params.pfcovars[:, :, 0] = PF.get_stats(particles)

            if linear:
                params.us[0] = MPC.mpc_var(params.pfmeans[:, 0], params.kfcovars[:, :, 0], horizon,
                                           A, numpy.matrix(B), b, aline, bline, e, params.QQ, params.RR, ysp,
                                           usp[0], limu, 1000.0, params.Q, k_squared, growvar)  # get controller input
            else:
                params.us[0] = MPC.mpc_var(params.pfmeans[:, 0]-b, params.kfcovars[:, :, 0], horizon,
                                           A, numpy.matrix(B), b, aline, bline, e, params.QQ, params.RR, ysp,
                                           usp[0], limu, 1000.0, params.Q, k_squared, growvar)  # get controller input
        else:
            params.us[0] = MPC.mpc_var(params.kfmeans[:, 0], params.kfcovars[:, :, 0], horizon,
                                       A, numpy.matrix(B), b, aline, bline, e, params.QQ, params.RR, ysp,
                                       usp[0], limu, 1000.0, params.Q, k_squared, growvar)  # get the controller input

        if numerical:
            kldiv[ndivcounter] = Auxiliary.kl(particles.x, particles.w,
                                              params.pfmeans[:, 0], params.pfcovars[:, :, 0], temp_states)
            basediv[ndivcounter] = Auxiliary.klbase(params.pfmeans[:, 0], params.pfcovars[:, :, 0], temp_states, nP)
            unidiv[ndivcounter] = Auxiliary.kluniform(params.pfmeans[:, 0], params.pfcovars[:, :, 0], temp_states, nP)
            klts[ndivcounter] = 0.0
            ndivcounter += 1
        status = True
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
            if pf:
                PF.pf_filter(particles, params.us[t - 1], params.ys2[:, t], state_noise_dist, meas_noise_dist, cstr_pf)
                params.pfmeans[:, t], params.pfcovars[:, :, t] = PF.get_stats(particles)

            if t % 10 == 0:
                if pf:
                    if linear:
                        params.us[t] = MPC.mpc_var(params.pfmeans[:, t], params.pfcovars[:, :, t], horizon,
                                                   A, numpy.matrix(B), b, aline, bline, e, params.QQ, params.RR, ysp,
                                                   usp[0], limu, 1000, params.Q, k_squared, growvar)  # controller input
                    else:
                        params.us[t] = MPC.mpc_var(params.pfmeans[:, t]-b, params.pfcovars[:, :, t], horizon,
                                                   A, numpy.matrix(B), b, aline, bline, e, params.QQ, params.RR, ysp,
                                                   usp[0], limu, 1000, params.Q, k_squared, growvar)  # controller input
                else:
                    params.us[t] = MPC.mpc_var(params.kfmeans[:, t], params.kfcovars[:, :, t], horizon,
                                               A, numpy.matrix(B), b, aline, bline, e, params.QQ, params.RR, ysp,
                                               usp[0], limu, 1000, params.Q, k_squared, growvar)  # get controller input

                if params.us[t] is None or numpy.isnan(params.us[t]):
                    status = False
                    break
            else:
                params.us[t] = params.us[t-1]

            if numerical and params.ts[t] in range(0, tend, 3):
                kldiv[ndivcounter] = Auxiliary.kl(particles.x, particles.w,
                                                  params.pfmeans[:, t], params.pfcovars[:, :, t], temp_states)
                basediv[ndivcounter] = Auxiliary.klbase(params.pfmeans[:, t], params.pfcovars[:, :, t], temp_states, nP)
                unidiv[ndivcounter] = Auxiliary.kluniform(params.pfmeans[:, t],
                                                          params.pfcovars[:, :, t],
                                                          temp_states, nP)
                klts[ndivcounter] = params.ts[t]
                ndivcounter += 1

        if not status:
            continue
        else:
            mciter += 1
        for i in range(len(params.kfmeans[0])):
            params.kfmeans[:, i] += b
            if linear:
                params.xs[:, i] += b
                params.ys2[:, i] += b

        if mcN > 1:
            xconcen[:, mciter] = params.xs[0, :]
            mcerrs[mciter] = Results.calc_error1(params.xs, ysp[0] + b[0])
            mcdists = Results.get_mc_res(params.xs, params.kfcovars, [aline, cline], mcdists, mciter, params.h)

        if mcN == 1:
            # Plot the results
            if numerical:
                Results.plot_kl_div(klts, kldiv, basediv, unidiv, False)
                Results.plot_kl_div(klts, kldiv, basediv, unidiv, True)
                print("The average divergence for the baseline is: ", 1.0 / len(klts) * sum(basediv))
                print("The average divergence for the approximation is: ", 1.0 / len(klts) * sum(kldiv))
                print("The average divergence for the uniform is: ", 1.0 / len(klts) * sum(unidiv))
            elif pf:
                Results.plot_tracking1(params.ts, params.xs, params.ys2, params.pfmeans, params.us, 2, ysp[0] + b[0])
                plt.savefig("/home/ex/Documents/CSC/report/results/Figure_8-25_python.pdf", bbox_inches="tight")
                Results.plot_ellipses2(params.ts, params.xs, params.pfmeans, params.pfcovars, [aline, cline],
                                       linsystems[1].op, True, -2.0 * numpy.log(1 - 0.9), plot_setting, "best")
                plt.savefig("/home/ex/Documents/CSC/report/results/Figure_8-26_python.pdf", bbox_inches="tight")
                Results.check_constraint(params.ts, params.xs, [aline, cline])
                Results.calc_error1(params.xs, ysp[0] + b[0])
                Results.calc_energy(params.us, 0.0)
            else:
                Results.plot_tracking1(params.ts, params.xs, params.ys2, params.kfmeans, params.us, 2, ysp[0]+b[0])
                plt.savefig("/home/ex/Documents/CSC/report/results/Figure_8-19_python.pdf", bbox_inches="tight")
                Results.plot_ellipses2(params.ts, params.xs, params.kfmeans, params.kfcovars, [aline, cline],
                                       linsystems[opoint].op, True, k_squared, plot_setting, "best")
                plt.savefig("/home/ex/Documents/CSC/report/results/Figure_8-20_python.pdf", bbox_inches="tight")
                Results.check_constraint(params.ts, params.xs, [aline, cline])
                Results.calc_error1(params.xs, ysp[0]+b[0])
                Results.calc_energy(params.us, 0.0)

            plt.show()

    if mcN != 1:
        print("The absolute MC average error is: ", sum(abs(mcerrs)) / mcN)
        if linear:
            if pf:
                numpy.savetxt("linmod_pf_var{}_mc2.csv".format(nine), xconcen, delimiter=",", fmt="%f")
            else:
                numpy.savetxt("linmod_kf_var{}_mc2.csv".format(nine), xconcen, delimiter=",", fmt="%f")
        else:
            if pf:
                numpy.savetxt("nonlinmod_pf_var{}_mc2.csv".format(nine), xconcen, delimiter=",", fmt="%f")
            else:
                numpy.savetxt("nonlinmod_kf_var{}_mc2.csv".format(nine), xconcen, delimiter=",", fmt="%f")

        nocount = len(mcdists[0]) - numpy.count_nonzero(mcdists[0])
        filteredResults = numpy.zeros([2, mcN - nocount])
        counter = 0
        for k in range(mcN):
            if mcdists[0, k] != 0.0:
                filteredResults[:, counter] = mcdists[:, k]
                counter += 1

        if linear:
            if pf:
                numpy.savetxt("linmod_pf_var{}_mc.csv".format(nine), filteredResults, delimiter=",",
                              fmt="%f")
            else:
                numpy.savetxt("linmod_kf_var{}_mc.csv".format(nine), filteredResults, delimiter=",",
                              fmt="%f")
        else:
            if pf:
                numpy.savetxt("nonlinmod_pf_var{}_mc.csv".format(nine), filteredResults, delimiter=",",
                              fmt="%f")
            else:
                numpy.savetxt("nonlinmod_kf_var{}_mc.csv".format(nine), filteredResults, delimiter=",",
                              fmt="%f")

