# Qualitative Analysis of the CSTR
# Nominal and bifurcation analysis

import numpy, sys, params, matplotlib as mpl, matplotlib.pyplot as plt, scipy.optimize
sys.path.append('../')
import src.Reactor as Reactor

tend = 150
params = params.Params(tend)
N = 100
Ts = numpy.linspace(200, 600, N) # temperature range
qrs1 = numpy.zeros(N) # heat removals
qrs2 = numpy.zeros(N) # heat removals
qrs3 = numpy.zeros(N) # heat removals
qgs1 = numpy.zeros(N) # heat generations

for k in range(N):
    qrs1[k] = params.cstr_model.QR(Ts[k], -906.)
    qrs2[k] = params.cstr_model.QR(Ts[k], 0.0)
    qrs3[k] = params.cstr_model.QR(Ts[k], 1145.0)
    qgs1[k] = params.cstr_model.QG(Ts[k])


mpl.rc("font", family="serif", serif="Computer Modern", size=12)
mpl.rc("text", usetex=True)

plt.figure(1)
q1, = plt.plot(Ts, qrs1, "b", linewidth=1)
q2, = plt.plot(Ts, qrs2, "g", linewidth=1)
q3, = plt.plot(Ts, qrs3, "r", linewidth=1)
opline1, = plt.plot(Ts, qgs1, "k", linewidth=1)
plt.legend([q1,q2,q3, opline1],["Q=-906 kJ/min","Q=0 kJ/min","Q=1145 kJ/min","Operating Curve"], loc="best")
plt.xlabel("Steady State Temperature [K]")
plt.ylabel("Heat Removal Rate [K/min]")
plt.ylim([0.0, 5])
plt.xlim([200, 600])
plt.show()

#Get the steady state points
xguess1 = [0.073, 493.0]
xguess2 = [0.21, 467.0]
xguess3 = [0.999, 310.0]
f = lambda x: params.cstr_model.reactor_func(x, 0.0)

xx1res = scipy.optimize.fsolve(f, xguess1)
# writecsv("ss1.csv", xx1res.zero)
xx2res = scipy.optimize.fsolve(f, xguess2)
# writecsv("ss2.csv", xx2res.zero)
xx3res = scipy.optimize.fsolve(f, xguess3)
# writecsv("ss3.csv", xx3res.zero)

print("Nominal Operating Points")
print("High Heat: ", xx1res)
print("Eigenvalues: ", numpy.linalg.eig(params.cstr_model.jacobian(xx1res ))[0])
print("Medium Heat: ", xx2res)
print("Eigenvalues: ", numpy.linalg.eig(params.cstr_model.jacobian(xx2res))[0])
print("Low Heat: ", xx3res)
print("Eigenvalues: ", numpy.linalg.eig(params.cstr_model.jacobian(xx3res))[0])

## Get the bifurcation points
# Low heat input
xguess1 = [0.999, 272]
xguess2 = [0.1089, 450]
flag = True
Q = -900.0
prevss1 = numpy.zeros(2)
prevss2 = numpy.zeros(2)
while flag:
    f = lambda x: params.cstr_model.reactor_func(x, Q)
    xx1res = scipy.optimize.fsolve(f, xguess1)
    xx2res = scipy.optimize.fsolve(f, xguess2)
    flag = sum(f(xx1res))  < 1e-8 and sum(f(xx2res))  < 1e-8
    if flag:
        prevss1 = xx1res
        prevss2 = xx2res
        Ql = Q
    Q = Q - 1
    if Q<-908:
        flag = False

print("Low heat: ", prevss1)
print("Eigenvalues: ", numpy.linalg.eig(params.cstr_model.jacobian(prevss1))[0])
print("*****")
print("Low heat: ", prevss2)
print("Eigenvalues: ", numpy.linalg.eig(params.cstr_model.jacobian(prevss2))[0])
print("*****")

# High heat input
xguess1 = [0.93, 370.0]
xguess2 = [0.0011, 570.0]
flag = True
Q = 1100.0
while flag:
    f = lambda x: params.cstr_model.reactor_func(x, Q)
    xx1res = scipy.optimize.fsolve(f, xguess1)
    xx2res = scipy.optimize.fsolve(f, xguess2)
    flag = sum(f(xx1res)) < 1e-8
    if flag:
        prevss1 = xx1res
        prevss2 = xx2res
        Qh = Q
    Q = Q + 1
    if Q > 1300:
        flag = False
print("High heat: ", prevss1)
print("Eigenvalues: ", numpy.linalg.eig(params.cstr_model.jacobian(prevss1))[0])
print("*****")
print("High heat: ", prevss2)
print("Eigenvalues: ", numpy.linalg.eig(params.cstr_model.jacobian(prevss2))[0])
print("*****")
