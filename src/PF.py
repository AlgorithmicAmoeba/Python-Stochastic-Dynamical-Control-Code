# Particle Filter module as explained in the paper "Novel approach to nonlinear/
# non-Gaussian Bayesian state estimation" by Gorden et al (1993).

import numpy
import warnings
import math
import copy


class Model:
    def __init__(self, f, g):
        self.f = f  # state transition model
        self.g = g  # state observation model


class Particles:
    """Implements a particle"""
    def __init__(self, x, w):
        self.x = x  # collection of particles
        self.w = w  # collection of particle weights


def init_pf(dist, nP, xN):
    """Initialise the particle filter.
    dist => a distribution from the package scipy.stats
    nX => number of states per particle.
    Return an array of nP particles."""

    particles = Particles(numpy.zeros([xN, nP]), numpy.zeros(nP))
    for p in range(nP):
        draw_x = dist.rsv()  # draw from the proposed prior
        particles.x[:, p] = draw_x
        particles.w[p] = 1./nP  # uniform initial weight

    return particles


def init_filter(particles, y, measure_dist, model):
    """Performs only the update step."""
    nX, N = particles.x.shape

    for p in range(N):
        particles.w[p] *= measure_dist.pdf(y - model.g(particles.x[:, p]))  # weight of each particle

    if abs(max(particles.w)) < 1e-8:
        warnings.warn("The particles all have very small weight...")

    particles.w = particles.w / sum(particles.w)
    for w in particles.w:
        if math.isnan(w):
            raise ValueError("Particles have become degenerate!")

    if number_effective_particles(particles) < N/2:
        particles = resample(particles)
    return particles


def roughen(particles):
    """Roughening the samples to promote diversity"""
    xN, N = particles.x.shape
    sig = numpy.zeros(xN)

    K = 0.2  # parameter...

    for k in range(xN):
        sig[k] = K*(max(particles.x[k, :]) - min(particles.x[k, :]))*N**(-1/xN)

    sigma = numpy.diag(sig**2)
    for p in range(N):
        jitter = numpy.random.multivariate_normal(numpy.zeros([len(sigma)]), sigma)
        particles.x[:, p] = particles.x[:, p] + jitter
    return particles


def resample(particles):
    N = len(particles.w)
    rs = numpy.random.choice(range(N), size=[N], p=particles.w)[0]  # draw N samples from weighted Categorical
    copy_particles = copy.copy(particles.x)
    for p in range(N):  # resample
        particles.x[:, p] = copy_particles[:, rs[p]]
        particles.w[p] = 1/N

    particles = roughen(particles)
    return particles


def number_effective_particles(particles):
    """Return the effective number of particles."""
    N = len(particles.w)
    num_eff = 0.0
    for p in range(N):
        num_eff += particles.w[p]**2

    return 1/num_eff


def pf_filter(particles, u, y, plantdist, measuredist, model):
    """Performs the state prediction step.
    plantnoise => distribution from whence the noise cometh
    measuredist => distribution from whence the plant uncertainty cometh"""

    nX, N = particles.x.shape

    for p in range(N):
        noise = plantdist.rsv()
        particles.x[:, p] = model.f(particles.x[:, p], u, noise)  # predict
        particles.w[p] *= measuredist.pdf(y - model.g(particles.x[:, p]))  # weight of each particle

    if abs(max(particles.w)) < 1e-8:
        warnings.warn("The particles all have very small weight...")

    particles.w = particles.w / sum(particles.w)
    for w in particles.w:
        if math.isnan(w):
            raise ValueError("Particles have become degenerate!")

    if number_effective_particles(particles) < N/2:
        particles = resample(particles)
    return particles


def get_stats(particles):
    """Return the Gaussian statistics of the particles"""
    mean = numpy.average(particles.x, weights=particles.w)
    cov = numpy.cov(particles.x, fweights=particles.w)
    return mean, cov


def predict(parts, u, plantdist, model):
    """Project the particles one step forward.
    NOTE: this overwrites parts therefore use a dummy variable!"""

    nX, nP = parts.shape
    for p in range(nP):
        noise = plantdist.rsv()
        parts[:, p] = model.f(parts[:, p], u, noise)  # predict
