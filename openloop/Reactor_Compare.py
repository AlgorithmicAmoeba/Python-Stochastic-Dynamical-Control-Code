# Compare the transient dynamics of two Reactors


import numpy, sys, params, matplotlib as mpl, matplotlib.pyplot as plt, scipy.optimize
sys.path.append('../')
import src.Reactor as Reactor

tend = 300
params = params.Params(tend)
initial_states = [0.5, 400]

xs1 = numpy.zeros([2, params.N])
xs2 = numpy.zeros([2, params.N])

us = numpy.zeros([params.N])
xs1[:,0] = initial_states
xs2[:,0] = initial_states
# Loop through the rest of time
for t in range(1,params.N):
    if params.ts[t] < 0.5:
        xs1[:, t] = params.cstr_model.run_reactor(xs1[:, t-1], us[t-1], params.h) # actual plant
        xs2[:, t] = params.cstr_model.run_reactor(xs2[:, t-1], us[t-1], params.h) # actual plant
    else:
        xs1[:, t] = params.cstr_model.run_reactor(xs1[:, t-1], us[t-1], params.h) # actual plant
        xs2[:, t] = params.cstr_model_broken.run_reactor(xs2[:, t-1], us[t-1], params.h) # actual plant



mpl.rc("font", family="serif", serif="Computer Modern", size=12)
mpl.rc("text", usetex=True)
skip = 50

# plt.figure(1) #
# x1, = plt.plot(xs1[0,:][:], xs1[1,:][:], "k", linewidth=3)
# x2, = plt.plot(xs2[0,:][:], xs2[1,:][:], "r--", linewidth=3)
# plt.ylabel("Temperature [K]")
# plt.xlabel(r"Concentration [kmol.m$^{-3}$]")

plt.figure(2) # Plot filtered results
plt.subplot(2,1,1)
x1, = plt.plot(params.ts, xs1[0,:], "k", linewidth=1)
# x2, = plot(params.ts, xs2[1,:]', "r--", linewidth=1)
plt.ylabel(r"C$_A$ [kmol.m$^{-3}$]")
plt.xlim([0, tend])
plt.locator_params(nbins=6)

plt.subplot(2,1,2)
x1, = plt.plot(params.ts, xs1[1,:], "k", linewidth=1)
# x2, = plot(params.ts, xs2[2,:]', "r--", linewidth=1)
plt.ylabel(r"T$_R$ [K]")
plt.xlabel("Time [min]")
plt.xlim([0, tend])
plt.locator_params(nbins=6)
plt.show()
