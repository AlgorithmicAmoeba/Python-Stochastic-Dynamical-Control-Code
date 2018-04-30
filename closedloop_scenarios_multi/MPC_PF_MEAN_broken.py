# Nonlinear plant model controlled with a linear MPC. The control goal is to steer
# the system to the unstead operating point. Deterministic contraints.

import numpy
import scipy.stats
import closedloop_scenarios_single.closedloop_params as params
import src.Results as Results
import src.LQR as LQR
import src.MPC as MPC
import src.PF as PF
import matplotlib.pyplot as plt

tend = 300
params = params.Params(tend)

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

# Set point
H = numpy.matrix([1, 0])  # only attempt to control the concentration
x_off, usp = LQR.offset(A, numpy.matrix(B), params.C2, H, numpy.matrix([ysp]))  # control offset
ysp = x_off
usp = numpy.array([usp])


def f(x, u, w):
    return params.cstr_model.run_reactor(x, u, params.h) + w


def g(x):
    return params.C2 @ x  # state observation


cstr_pf = PF.Model(f, g)

# Initialise the PF
nP = 200  # number of particles.
prior_dist = scipy.stats.multivariate_normal(mean=init_state, cov=params.init_state_covar)  # prior distribution
particles = PF.init_pf(prior_dist, nP, 2)  # initialise the particles
state_noise_dist = scipy.stats.multivariate_normal(cov=params.Q)  # state distribution
meas_noise_dist = scipy.stats.multivariate_normal(cov=params.R2)  # measurement distribution

# First time step of the simulation
params.xs[:, 0] = init_state  # set simulation starting point to the random initial state
params.ys2[:, 0] = params.C2 @ params.xs[:, 0] + meas_noise_dist.rvs()  # measure from actual plant
PF.init_filter(particles, params.ys2[:, 0], meas_noise_dist, cstr_pf)
params.pfmeans[:, 0], params.pfcovars[:, :, 0] = PF.get_stats(particles)

# Setup MPC
horizon = 150
# add state constraints
aline = 10.  # slope of constraint line ax + by + c = 0
cline = -420.0  # negative of the y axis intercept
bline = 1.0

params.us[0] = MPC.mpc_mean(params.pfmeans[:, 0]-b, horizon, A, numpy.matrix(B), b, aline, bline, cline, params.QQ, params.RR, ysp, usp[0], 15000.0, 1000.0)  # get the controller input
for t in range(1, params.N):
    d = numpy.zeros(2)
    if params.ts[t] < 100:
        params.xs[:, t] = params.cstr_model.run_reactor(params.xs[:, t-1], params.us[t-1], params.h) + state_noise_dist.rvs()  # actual plant
        xtemp = params.xs[:, t-1] - b
        d = params.cstr_model.run_reactor(params.xs[:, t-1], params.us[t-1], params.h) - (A @ xtemp + B * params.us[t-1] + b)
    else:
        params.xs[:, t] = params.cstr_model_broken.run_reactor(params.xs[:, t-1], params.us[t-1], params.h) + state_noise_dist.rvs()  # actual plant
        xtemp = params.xs[:, t-1] - b
        d = params.cstr_model_broken.run_reactor(params.xs[:, t-1], params.us[t-1], params.h) - (A @ xtemp + B * params.us[t-1] + b)

    params.ys2[:, t] = params.C2 @ params.xs[:, t] + meas_noise_dist.rvs()  # measure from actual plant
    PF.pf_filter(particles, params.us[t-1], params.ys2[:, t], state_noise_dist, meas_noise_dist, cstr_pf)
    params.pfmeans[:, t], params.pfcovars[:, :, t] = PF.get_stats(particles)
    if t % 1 == 0:
        params.us[t] = MPC.mpc_mean(params.pfmeans[:, t]-b, horizon, A, numpy.matrix(B), b, aline, bline, cline, params.QQ, params.RR, ysp, usp[0], 15000.0, 1000.0, d)
    else:
        params.us[t] = params.us[t-1]
    if params.us[t] is None or numpy.isnan(params.us[t]):
        break


# # Plot the results
Results.plot_tracking1(params.ts, params.xs, params.ys2, params.pfmeans, params.us, 2, ysp[0]+b[0])
Results.plot_ellipses2(params.ts, params.xs, params.pfmeans, params.pfcovars, [aline, cline], [linsystems[2].op[1], 422.6], True, 4.6052, 1, "upper right")
Results.check_constraint(params.ts, params.xs, [aline, cline])
Results.calc_error1(params.xs, ysp[0]+b[0])
Results.calc_energy(params.us, params.h)
plt.show()
