# Inference using one linear model measuring only temperature


import matplotlib.pyplot as plt
import numpy
import sys

sys.path.append('../')
import openloop.params
import src.LLDS as LLDS
import src.Results as Results

tend = 50

params = openloop.params.Params(tend)
init_state = [0.5, 400]

# Specify the linear model
linsystems = params.cstr_model.get_nominal_linear_systems(params.h)
opoint = 1  # which nominal model to use
A = linsystems[opoint].A
B = linsystems[opoint].B
b = linsystems[opoint].b

lin_cstr = LLDS.LLDS(A, B, params.C1, params.Q, params.R1)  # KF object

# Plant initialisation
params.xs[:, 0] = init_state
params.linxs[:, 0] = init_state - b

# Simulate plant
state_noise_dist = numpy.random.multivariate_normal(numpy.zeros([len(params.Q)]), params.Q)
meas_noise_dist = numpy.random.normal(0, numpy.sqrt(params.R1[0]))
params.ys1[0] = params.C1 @ params.xs[:, 0] + meas_noise_dist  # measure from actual plant

# Filter setup
kfmeans = numpy.zeros([2, params.N])
kfcovars = numpy.zeros([2, 2, params.N])
init_mean = init_state - b

# First time step
kfmeans[:, 0], kfcovars[:, :, 0] = lin_cstr.init_filter(init_mean, params.init_state_covar, params.ys1[0]-b[1])

for t in range(1, params.N):
    state_noise_dist = numpy.random.multivariate_normal(numpy.zeros([len(params.Q)]), params.Q)
    params.xs[:, t] = params.cstr_model.run_reactor(params.xs[:, t-1], params.us[t-1], params.h) + state_noise_dist
    meas_noise_dist = numpy.random.normal(0, numpy.sqrt(params.R1[0]))
    params.ys1[t] = params.C1 @ params.xs[:, t] + meas_noise_dist  # measured from actual plant
    params.linxs[:, t], temp = lin_cstr.step(params.linxs[:, t-1], params.us[t-1])
    temp = lin_cstr.step_filter(kfmeans[:, t-1], kfcovars[:, :, t-1], params.us[t-1], params.ys1[t] - b[1])
    kfmeans[:, t], kfcovars[:, :, t] = temp

for i in range(len(params.linxs[0])):
    params.linxs[:, i] += b
    kfmeans[:, i] += b

# Plot results

Results.plot_ellipses1(params.ts, params.xs, kfmeans, kfcovars, "Kalman Filter", "upper right")

Results.plot_tracking(params.ts, params.xs, params.ys1, kfmeans, params.us, 1)

plt.show()
avediff = Results.calc_error(params.xs, kfmeans)
