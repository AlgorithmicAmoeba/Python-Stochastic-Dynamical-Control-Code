# Implement the augmented switching dynamical system

import numpy
import scipy.stats
import openloop.params as params
import src.Results as Results
import src.RBPF as RBPF
import matplotlib.pyplot as plt

tend = 150
params = params.Params(tend)

init_state = numpy.array([0.5, 400])  # initial state

# Divide state space into sectors: n by m
nX = 2  # rows
nY = 2  # cols
xspace = numpy.array([0.0, 1.0])
yspace = numpy.array([250, 650])

linsystems = params.cstr_model.get_nominal_linear_systems(params.h)

models, _ = RBPF.setup_rbpf(linsystems, params.C2, params.Q, params.R2)
A = numpy.array([[0.99, 0.01, 0.00],
                 [0.01, 0.98, 0.01],
                 [0.00, 0.01, 0.99]])
numModels = len(models)

nP = 500
sguess = RBPF.get_initial_switches(init_state, linsystems)
particles = RBPF.init_rbpf(sguess, init_state, params.init_state_covar, 2, nP)

switchtrack = numpy.zeros([len(linsystems), params.N])
maxtrack = numpy.zeros([len(linsystems), params.N])
smoothedtrack = numpy.zeros([len(linsystems), params.N])

state_noise_dist = scipy.stats.multivariate_normal(cov=params.Q)
meas_noise_dist = scipy.stats.multivariate_normal(cov=params.R2)

params.xs[:, 0] = init_state
params.ys2[:, 0] = params.C2 @ params.xs[:, 0] + meas_noise_dist.rvs()  # measured from actual plant

particles = RBPF.init_filter(particles, 0.0, params.ys2[:, 0], models)
params.rbpfmeans[:, 0], params.rbpfcovars[:, :, 0] = RBPF.get_ave_stats(particles)
# rbpfmeans[:, 0], rbpfcovars[:, :, 0] = RBPF.getMLStats(particles)

for k in range(len(linsystems)):
        loc = numpy.where(particles.ss == k)[0]
        s = 0
        for l in loc:
            s += particles.ws[l]

        switchtrack[k, 0] = s

maxtrack[:, 0] = RBPF.get_max_track(particles, numModels)
smoothedtrack[:, 0] = RBPF.smoothed_track(numModels, switchtrack, 1, 10)

# Loop through the rest of time
for t in range(1, params.N):
    params.xs[:, t] = params.cstr_model.run_reactor(params.xs[:, t-1], params.us[t-1], params.h)
    params.xs[:, t] += state_noise_dist.rvs()
    params.ys2[:, t] = params.C2 @ params.xs[:, t] + meas_noise_dist.rvs()  # measured from actual plant
    particles = RBPF.rbpf_filter(particles, params.us[t-1], params.ys2[:, t], models, A)
    # rbpfmeans[:, t], rbpfcovars[:,:, t] = RBPF.getMLStats(particles)
    params.rbpfmeans[:, t], params.rbpfcovars[:, :, t] = RBPF.get_ave_stats(particles)

    for k in range(len(linsystems)):
        loc = numpy.where(particles.ss == k)[0]
        s = 0
        for l in loc:
            s += particles.ws[l]
        switchtrack[k, t] = s

    maxtrack[:, t] = RBPF.get_max_track(particles, numModels)
    smoothedtrack[:, t] = RBPF.smoothed_track(numModels, switchtrack, t, 10)

# Plot results
Results.plot_state_space_switch(linsystems, params.xs)
Results.plot_switch_selection(numModels, switchtrack, params.ts, True)
# Results.plotSwitchSelection(numModels, maxtrack, ts, false)
# Results.plotSwitchSelection(numModels, smoothedtrack, ts, false)
Results.plot_tracking(params.ts, params.xs, params.ys2, params.rbpfmeans, params.us, 2)
Results.calc_error(params.xs, params.rbpfmeans)
plt.show()
print("DONE")
