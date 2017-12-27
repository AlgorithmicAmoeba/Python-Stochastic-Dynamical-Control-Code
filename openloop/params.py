# All the common model parameters go here

# All the required modules
import numpy, sys
sys.path.append('../')
import src.Reactor as Reactor


class Params:
    # Extend the Base Library
    def convert_to_scalar(self, x):
      return x[1]
      
    def __init__(self, tend):
        self.tend = tend

        # Specify the nonlinear model
        self.cstr_model = Reactor.Reactor()

        # Specify the nonlinear model of the broken plant
        self.cstr_model_broken = Reactor.Reactor(k0 = 0.1*72.0e7)
        # Discretise the system
        self.h = 0.1 # time discretisation
        self.ts = numpy.arange(0,self.tend+self.h,self.h)
        self.N = len(self.ts)
        self.xs = numpy.zeros([2, self.N]) # nonlinear plant
        self.linxs = numpy.zeros([2, self.N]) # linear plant
        self.xsnofix = numpy.zeros([2, self.N]) # broken plant
        self.ys1 = numpy.zeros([self.N]) # only measure temperature
        self.ys2 = numpy.zeros([2, self.N]) # measure both concentration and temperature
        self.ys2nofix = numpy.zeros([2, self.N])
        self.us = numpy.zeros([self.N]) # controller input
        self.usnofix = numpy.zeros([self.N])

        self.init_state_covar = numpy.eye(2) # prior covariance
        self.init_state_covar[0][0] = 1e-3
        self.init_state_covar[1][1] = 4.

        self.pfmeans = numpy.zeros([2, self.N]) # Particle Filter means
        self.pfcovars = numpy.zeros([2,2, self.N]) # Particle Filter covariances (assumed Gaussian)
        self.rbpfmeans = numpy.zeros([2, self.N]) # RBPF means
        self.rbpfcovars = numpy.zeros([2,2, self.N]) # RBPF covariances
        self.kfmeans = numpy.zeros([2, self.N]) # Kalman Filter means
        self.kfcovars = numpy.zeros([2,2, self.N]) # Kalman Filter covariances
        self.spfmeans  = numpy.zeros([2, self.N]) # Kalman Filter means
        self.spfcovars = numpy.zeros([2,2, self.N])

        # Noise settings
        self.Q = numpy.eye(2) # plant noise
        self.Q[0][0] = 1e-06
        self.Q[1][1] = 0.1

        self.R1 = numpy.eye(1)*10.0 # measurement noise (only temperature)
        self.R2 = numpy.eye(2) # measurement noise (both concentration and temperature)
        self.R2[0][0] = 1e-3
        self.R2[1][1] = 10.0

        # Measurement settings
        self.C2 = numpy.eye(2) # we measure both concentration and temperature
        self.C1 = [0.0, 1.0] # we measure only temperature

        # Controller settings (using quadratic cost function)
        self.QQ = numpy.zeros([2, 2])
        self.QQ[0][0] = 10000.0 # due to the magnitude of the concentration
        self.RR = 0.000001 #

    # seed the random number generator
    # srand(745) # good for KF, SPF
    # srand(3265) # good for RBPF
