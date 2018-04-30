# Inference using two nonlinear models measuring only temperature

import numpy
import scipy.stats
import openloop.params as params
import src.Results as Results
import src.RBPF as RBPF
import src.SPF as SPF
import matplotlib.pyplot as plt

tend = 300
params = params.Params(tend)
# srand(8745)

init_state = numpy.array([0.55, 450])

A = numpy.array([[0.99, 0.01],
                 [0.01, 0.99]])


def fun1(x, u, w):
    return params.cstr_model.run_reactor(x, u, params.h) + w


def fun2(x, u, w):
    return params.cstr_model_broken.run_reactor(x, u, params.h) + w


def gs(x):
    return params.C2 @ x


F = [fun1, fun2]
G = [gs, gs]
numSwitches = 2

ydists = numpy.array([scipy.stats.multivariate_normal(cov=params.R2), scipy.stats.multivariate_normal(cov=params.R2)])
xdists = numpy.array([scipy.stats.multivariate_normal(cov=params.Q), scipy.stats.multivariate_normal(cov=params.Q)])
cstr_filter = SPF.Model(F, G, A, xdists, ydists)

nP = 500
xdist = scipy.stats.multivariate_normal(mean=init_state, cov=params.init_state_covar)
sdist = [0.9, 0.1]
particles = SPF.init_spf(xdist, sdist, nP, 2)

switchtrack = numpy.zeros([2, params.N])
maxtrack = numpy.zeros([numSwitches, params.N])
smoothedtrack = numpy.zeros([numSwitches, params.N])

state_noise_dist = scipy.stats.multivariate_normal(cov=params.Q)
meas_noise_dist = scipy.stats.multivariate_normal(cov=params.R2)

params.xs[:, 0] = init_state
params.xsnofix[:, 0] = init_state

params.ys2[:, 0] = params.C2 @ params.xs[:, 0] + meas_noise_dist.rvs()  # measured from actual plant
SPF.init_filter(particles, params.ys2[:, 0], cstr_filter)

for k in range(numSwitches):
    switchtrack[k, 0] = numpy.sum(particles.w[numpy.where(particles.s == k)[0]])

maxtrack[:, 0] = SPF.get_max_track(particles, numSwitches)
smoothedtrack[:, 0] = RBPF.smoothed_track(numSwitches, switchtrack, 1, 10)

params.spfmeans[:, 0], params.spfcovars[:, :, 0] = SPF.get_stats(particles)

# Loop through the rest of time
for t in range(1, params.N):
    random_element = state_noise_dist.rvs()
    if params.ts[t] < 50.0:
        params.xs[:, t] = params.cstr_model.run_reactor(params.xs[:, t-1], params.us[t-1], params.h) + random_element
        params.xsnofix[:, t] = params.cstr_model.run_reactor(params.xsnofix[:, t-1], params.us[t-1], params.h)
        params.xsnofix[:, t] += random_element  # actual plant
    else:
        params.xs[:, t] = params.cstr_model_broken.run_reactor(params.xs[:, t-1], params.us[t-1], params.h)
        params.xs[:, t] += random_element
        params.xsnofix[:, t] = params.cstr_model.run_reactor(params.xsnofix[:, t-1], params.us[t-1], params.h)
        params.xsnofix[:, t] += random_element  # actual plant

    params.ys2[:, t] = params.C2 @ params.xs[:, t] + meas_noise_dist.rvs()  # measured from actual plant
    SPF.spf_filter(particles, params.us[t-1], params.ys2[:, t], cstr_filter)
    params.spfmeans[:, t], params.spfcovars[:, :, t] = SPF.get_stats(particles)

    for k in range(2):
        ind = numpy.where(particles.s == k)[0]
        if ind.shape[0] != 0:
            switchtrack[k, t] = numpy.sum(particles.w[ind])

    maxtrack[:, t] = SPF.get_max_track(particles, numSwitches)
    smoothedtrack[:, t] = RBPF.smoothed_track(numSwitches, switchtrack, t, 10)


# Plot results
Results.plot_switch_selection(numSwitches, switchtrack, params.ts, True)
Results.plot_switch_selection(numSwitches, maxtrack, params.ts, False)
Results.plot_tracking_break(params.ts, params.xs, params.xsnofix, params.ys2, params.spfmeans, 2)
Results.calc_error(params.xs, params.spfmeans)
plt.show()
