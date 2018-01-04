# PF inference using the full nonlinear model

import openloop.params
import numpy
import src.PF as PF
import scipy.stats
import src.Results as Results
import matplotlib.pyplot as plt

tend = 50
params = openloop.params.Params(tend)

init_state = numpy.array([0.5, 400])  # initial state


def f(x, u, w):
    return params.cstr_model.run_reactor(x, u, params.h) + w  # transition function


def g(x):
    return params.C2 @ x  # state observation


cstr_pf = PF.Model(f, g)

# Initialise the PF
nP = 200  # number of particles.
prior_dist = scipy.stats.multivariate_normal(mean=init_state, cov=params.init_state_covar)  # prior distribution
particles = PF.init_pf(prior_dist, nP, 2)  # initialise the particles
state_noise_dist = scipy.stats.multivariate_normal(cov=params.Q)  # state distribution
meas_noise_dist = scipy.stats.multivariate_normal(cov=params.R2)  # measurement distribution

# Time step 1
params.xs[:, 0] = init_state
params.ys2[:, 0] = params.C2 @ params.xs[:, 0] + meas_noise_dist.rvs()  # measured from actual plant
particles = PF.init_filter(particles, params.ys2[:, 0], meas_noise_dist, cstr_pf)
params.pfmeans[:, 0], params.pfcovars[:, :, 0] = PF.get_stats(particles)

# Loop through the rest of time
for t in range(1, params.N):
    params.xs[:, t] = params.cstr_model.run_reactor(params.xs[:, t - 1], params.us[t - 1], params.h)
    params.xs[:, t] += state_noise_dist.rvs()  # actual plant
    params.ys2[:, t] = params.C2 @ params.xs[:, t] + meas_noise_dist.rvs()  # measured from actual plant
    particles = PF.pf_filter(particles, params.us[t-1], params.ys2[:, t], state_noise_dist, meas_noise_dist, cstr_pf)
    params.pfmeans[:, t], params.pfcovars[:, :, t] = PF.get_stats(particles)

# Plot results

Results.plot_ellipses1(params.ts, params.xs, params.pfmeans, params.pfcovars, "upper right")

Results.plot_tracking(params.ts, params.xs, params.ys2, params.pfmeans, params.us, 2)

Results.calc_error(params.xs, params.pfmeans)

plt.show()
