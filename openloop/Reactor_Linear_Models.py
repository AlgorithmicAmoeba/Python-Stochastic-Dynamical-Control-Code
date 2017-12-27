# Linearisation Procedure
import numpy, sys, params, matplotlib as mpl, matplotlib.pyplot as plt, scipy.optimize
sys.path.append('../')
import src.Reactor as Reactor

tend = 50
params = params.Params(tend)

xspace = [0.0, 1.0]
yspace = [250, 650]
linsystems = params.cstr_model.getLinearSystems_randomly(0, xspace, yspace, params.h)


mpl.rc("font", family="serif", serif="Computer Modern", size=12)
mpl.rc("text", usetex=True)
plt.figure(1)
k=2 # set which operating point to use
# also remember to change +- on line 47 and the SS points on lines 75-81
nDD = 2
x1 = 0 # legend var
x2 = 0 # legend var
x3 = 0 # legend var
for dd in range(nDD): # only loop through
    initial_states = linsystems[k].op + (numpy.random.random([2])*2-1)*[0.01, 10]*(dd+1)

    N = len(params.ts)
    xs = numpy.zeros([2, N])
    linxs = numpy.zeros([2, N])
    xs[:,0] = initial_states
    linxs[:,0] = initial_states - linsystems[k].b
    # Loop through the rest of time
    for t in range(1, N):
        xs[:, t] = params.cstr_model.run_reactor(xs[:, t-1], 0.0, params.h) # actual plant
        linxs[:, t] = numpy.matmul(linsystems[k].A, linxs[:, t-1]) + linsystems[k].B
    
    for i in range(len(linxs[0])):
        linxs[:,i] += linsystems[k].b

    
        
    plt.subplot(nDD, 1, dd+1)
    x1, = plt.plot(xs[0], xs[1], "k", linewidth=3)
    x2, = plt.plot(linxs[0], linxs[1], "r--", linewidth=3)
    plt.plot(xs[0][0], xs[1][0], "ko", markersize=10, markeredgewidth = 4)
    plt.plot(xs[0][-1], xs[1][-1], "kx", markersize=10, markeredgewidth = 4)
    plt.plot(linxs[0][0], linxs[1][0], "ro", markersize=10, markeredgewidth = 4)
    plt.plot(linxs[0][-1], linxs[1][-1], "rx", markersize=10, markeredgewidth = 4)
    plt.ylabel(r"T$_R$ [K]")
    plt.locator_params(nbins=6)

    ## Comment out as necessary!
    if k==0:
        ss1 = [0.0097, 508.0562]
        x3, = plt.plot(ss1[0], ss1[1], "gx", markersize=10, markeredgewidth = 4)
    elif k==1:
        ss2 = [0.4893, 412.1302]
        x3, = plt.plot(ss2[0], ss2[1], "gx", markersize=10, markeredgewidth = 4)
    else:
        ss3 = [0.9996, 310.0709]
        x3, = plt.plot(ss3[0], ss3[1], "gx", markersize=10, markeredgewidth = 4)

plt.legend([x1, x2, x3],["Nonlinear model","Linear model","Operating point"], loc="best")
plt.xlabel(r"C$_A$ [kmol.m$^{-3}$]")
plt.locator_params(nbins=6)
plt.show()
