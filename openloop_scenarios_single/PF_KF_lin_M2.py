# Inference using a linear model in the PF compared to the KF (which implicitly uses a linear model)
# NOTE: don't run for too long because the linear model is unstable!

import openloop.params
import numpy
import src.PF as PF
import scipy.stats
import src.Results as Results
import matplotlib.pyplot as plt
import src.LLDS as LLDS

tend = 20
params = openloop.params.Params(tend)

init_state = numpy.array([0.5, 400])

# Specify the linear model
lin_systems = params.cstr_model.get_nominal_linear_systems(params.h)
o_point = 1
A = lin_systems[o_point].A
B = lin_systems[o_point].B
b = lin_systems[o_point].b

lin_cstr = LLDS.LLDS(A, B, params.C2, params.Q, params.R2)  # KF object

# Setup the PF


def f(x, u, w):
    return A @ x + B * u + w


def g(x):
    return params.C2 @ x  # state observation


cstr_pf = PF.Model(f, g)
nP = 500  # number of particles.

# Initialise the PFs
init_pf_dist = scipy.stats.multivariate_normal(mean=init_state-b, cov=params.init_state_covar)  # prior distribution
particles = PF.init_pf(init_pf_dist, nP, 2)  # initialise the particles

state_noise_dist = scipy.stats.multivariate_normal(cov=params.Q)  # state distribution
meas_noise_dist = scipy.stats.multivariate_normal(cov=params.R2)  # measurement distribution

pfmeans = numpy.zeros([2, params.N])
pfcovars = numpy.zeros([2, 2, params.N])
kfmeans = numpy.zeros([2, params.N])
kfcovars = numpy.zeros([2, 2, params.N])

# Time step 1
params.xs[:, 0] = init_state
params.ys2[:, 0] = params.C2 @ params.xs[:, 0] + meas_noise_dist.rvs()  # measured from actual plant
particles = PF.init_filter(particles, params.ys2[:, 0]-b, meas_noise_dist, cstr_pf)
pfmeans[:, 0], pfcovars[:, :, 0] = PF.get_stats(particles)
kfmeans[:, 0], kfcovars[:, :, 0] = lin_cstr.init_filter(init_state-b, params.init_state_covar, params.ys2[:, 0]-b)

# Loop through the rest of time
for t in range(1, params.N):
    params.xs[:, t] = A @ (params.xs[:, t-1]-b) + B * params.us[t-1] + b + state_noise_dist.rvs()  # actual plant
    params.ys2[:, t] = params.C2 @ params.xs[:, t] + meas_noise_dist.rvs()  # measured from actual plant
    particles = PF.pf_filter(particles, params.us[t-1], params.ys2[:, t]-b, state_noise_dist, meas_noise_dist, cstr_pf)
    pfmeans[:, t], pfcovars[:, :, t] = PF.get_stats(particles)
    temp = lin_cstr.step_filter(kfmeans[:, t-1], kfcovars[:, :, t-1], params.us[t-1], params.ys2[:, t]-b)
    kfmeans[:, t], kfcovars[:, :, t] = temp

for i in range(len(pfmeans[0])):
    pfmeans[:, i] += b
    kfmeans[:, i] += b

# Plot Results
Results.plot_ellipse_comp(pfmeans, pfcovars, kfmeans, kfcovars, params.xs, params.ts)

Results.plot_tracking_two_filters(params.ts, params.xs, params.ys2, pfmeans, kfmeans)

print("For the Kalman Filter:")
Results.calc_error(params.xs, kfmeans)
print("For the Particle Filter:")
Results.calc_error(params.xs, pfmeans)

plt.show()
