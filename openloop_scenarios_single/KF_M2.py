# Inference using one linear model measuring only temperature


import sys, numpy, matplotlib as mpl, matplotlib.pyplot as plt
sys.path.append('../')
import openloop.params
import src.Reactor as Reactor, src.LLDS as LLDS, src.Results as Results

tend = 50

params = openloop.params.Params(tend)
init_state = [0.5, 400]

# Specify the linear model
linsystems = params.cstr_model.getNominalLinearSystems(params.h)
opoint = 1 # which nominal model to use
A = linsystems[opoint].A
B = linsystems[opoint].B
b = linsystems[opoint].b

lin_cstr = LLDS.llds(A, B, params.C2, params.Q, params.R2) # KF object

# Plant initialisation
params.xs[:,0] = init_state
params.linxs[:, 0] = init_state - b

# Simulate plant
state_noise_dist = numpy.random.multivariate_normal(numpy.zeros([len(params.Q)]), params.Q)
meas_noise_dist = numpy.random.multivariate_normal(numpy.zeros([len(lin_cstr.R)]), lin_cstr.R)
params.ys2[:,0] = numpy.matmul(params.C2, params.xs[:,0]) + meas_noise_dist # measure from actual plant

# Filter setup
kfmeans = numpy.zeros([2, params.N])
kfcovars = numpy.zeros([2,2, params.N])
init_mean = init_state - b

# First time step
kfmeans[:, 0], kfcovars[:,:, 0] = lin_cstr.init_filter(init_mean, params.init_state_covar, params.ys2[:,0]-b)

for t in range(1,params.N):
    state_noise_dist = numpy.random.multivariate_normal(numpy.zeros([len(params.Q)]), params.Q)
    params.xs[:, t] = params.cstr_model.run_reactor(params.xs[:, t-1], params.us[t-1], params.h) + state_noise_dist# actual plant
    meas_noise_dist = numpy.random.multivariate_normal(numpy.zeros([len(lin_cstr.R)]), lin_cstr.R)
    params.ys2[:,t] = numpy.matmul(params.C2, params.xs[:, t]) + meas_noise_dist # measured from actual plant
    params.linxs[:, t], temp = lin_cstr.step(params.linxs[:, t-1], params.us[t-1])
    kfmeans[:, t], kfcovars[:,:, t] = lin_cstr.step_filter(kfmeans[:, t-1], kfcovars[:,:, t-1], params.us[t-1], params.ys2[:,t] - b)

for i in range(len(params.linxs[0])):
    params.linxs[:,i]+= b
    kfmeans[:,i]+= b

# Plot results

Results.plotEllipses1(params.ts, params.xs, kfmeans, kfcovars, "Kalman Filter", "lower left")

Results.plotTracking(params.ts, params.xs, params.ys2, kfmeans, params.us, 2)

plt.show()
avediff = Results.calcError(params.xs, kfmeans)
