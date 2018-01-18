# Test the Particle Filter.

import src.PF as PF
import src.Reactor as Reactor
import pandas
import numpy
import scipy.stats
import pathlib

A_path = pathlib.Path("A.csv")
if A_path.is_file():
    A = pandas.read_csv("A.csv", header=None).as_matrix()
    B = pandas.read_csv("B.csv", header=None).as_matrix().T[0]
    b = pandas.read_csv("b1.csv", header=None).as_matrix().T[0]
    kfmeans = pandas.read_csv("KFmeans.csv", header=None).as_matrix()
else:
    A = pandas.read_csv("test/A.csv", header=None).as_matrix()
    B = pandas.read_csv("test/B.csv", header=None).as_matrix().T[0]
    b = pandas.read_csv("test/b1.csv", header=None).as_matrix().T[0]
    kfmeans = pandas.read_csv("test/KFmeans.csv", header=None).as_matrix()

cstr_model = Reactor.Reactor(V=5.0, R=8.314, CA0=1.0, TA0=310.0, dH=-4.78e4,
                             k0=72.0e7, E=8.314e4, Cp=0.239, rho=1000.0, F=100e-3)


init_state = numpy.array([0.50, 400])
h = 0.01  # time discretisation
tend = 20.0  # end simulation time
ts = [x/100 for x in range(2001)]
N = len(ts)
xs = numpy.zeros([2, N])
xs[:, 0] = init_state
ys = numpy.zeros([2, N])  # only one measurement


C = numpy.eye(2)
Q = numpy.eye(2)  # plant mismatch/noise
Q[0, 0] = 1e-6
Q[1, ] = 4.
R = numpy.eye(2)
R[0, 0] = 1e-4
R[1, 1] = 10.0  # measurement noise


def f(x, u, w):
    return A @ x + B * u + b + w


def g(x):
    return C @ x  # state observation


cstr_pf = PF.Model(f, g)

# Initialise the PF
nP = 50  # number of particles.
init_state_mean = init_state  # initial state mean
init_state_covar = numpy.eye(2) * 1e-6  # initial covariance
init_state_covar[1, 1] = 2.0
init_dist = scipy.stats.multivariate_normal(mean=init_state_mean, cov=init_state_covar)  # prior distribution
state_covar = numpy.eye(2)  # state covariance
state_covar[0, 0] = 1e-4
state_covar[1, 1] = 4.
state_dist = scipy.stats.multivariate_normal(cov=state_covar)  # state distribution
meas_covar = numpy.eye(2)
meas_covar[0, 0] = 1e-4
meas_covar[1, 1] = 10.
meas_dist = scipy.stats.multivariate_normal(cov=meas_covar)  # measurement distribution


def test_filter():
    particles = PF.init_pf(init_dist, nP, 2)  # initialise the particles
    fmeans = numpy.zeros([2, N])
    fcovars = numpy.zeros([2, 2, N])
    # Time step 1
    xs[:, 0] = init_state
    ys[:, 0] = C @ xs[:, 0] + meas_dist.rvs()  # measured from actual plant
    particles = PF.init_filter(particles, ys[:, 0], meas_dist, cstr_pf)
    fmeans[:, 0], fcovars[:, :, 0] = PF.get_stats(particles)
    # Loop through the rest of time
    for t in range(1, N):
        xs[:, t] = cstr_model.run_reactor(xs[:, t-1], 0.0, h)  # actual plant
        ys[:, t] = C @ xs[:, t] + meas_dist.rvs()  # measured from actual plant
        particles = PF.pf_filter(particles, 0.0, ys[:, t], state_dist, meas_dist, cstr_pf)
        fmeans[:, t], fcovars[:, :, t] = PF.get_stats(particles)

    tol = 20.0
    assert (abs(fmeans-kfmeans)).max() < tol


if __name__ == '__main__':
    test_filter()
