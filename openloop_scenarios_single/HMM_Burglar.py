# Hidden Markov Model
# h - h -> ...
# |   |
# o   o
# Problem taken from:
# Bayesian Reasoning and Machine Learning
# by David Barber
# Website: http://www0.cs.ucl.ac.uk/staff/d.barber/brml/
# Chapter 23, Example 23.3: Localisation Problem
import sys, numpy, matplotlib as mpl, matplotlib.pyplot as plt
sys.path.append('../')
import src.HMM as HMM
import src.Burglar as Burglar


T = 10
n = 5

house = Burglar.House(n)
hmm = house.createHMM() # create the hidden markov model

movements = numpy.zeros([n,n,T])
locs = numpy.zeros([T], dtype=numpy.int)
observations = numpy.zeros([T], dtype=numpy.int)
noises = numpy.zeros([2,2,T])

# Initial distribution
# initial = normalise(rand(n*n)) # no idea where the burglar is - not good
initial = numpy.zeros([n*n]) # know the burglar is near the entrance of the house
# initial[1:2*n] = 1.0/(2.*n) # two column initial guess
initial[::n] = 1/n # one column initial guess


# Measurement and actual movements
for t in range(T):
    locs[t] = house.getLocation()
    movements[:,:,t] = house.floor
    observations[t] = numpy.random.choice(range(4), size = [1], p = hmm.ep[:,locs[t]])[0]
    house.move()
    if observations[t] == 0:
        noises[:,:,t] = [[1,1],[1,1]]
    elif observations[t] == 1:
        noises[:,:,t] = [[1,0],[1,0]]
    elif observations[t] == 2:
        noises[:,:,t] = [[0,1],[0,1]]
    else:
        noises[:,:,t] = [[0,0],[0,0]]

## Inference
# Forward Filter
filter_ = hmm.forward(initial, observations)

# Smoothing
fbs = numpy.zeros([ len(initial),  len(observations)])
for k in range( len(observations)):
    fbs[:, k] = hmm.smooth(initial, observations, k)

# Viterbi
vtb = hmm.viterbi_dp(initial, observations)
mlmove = numpy.zeros([n,n,T]) # construct floor matrices showing the viterbi path
for k in range( len(observations)):
    temp = numpy.zeros([n,n])
    temp[vtb[k]//n, vtb[k]%n] = 1
    mlmove[:,:, k] = temp

# Prediction
predmove = numpy.zeros([n, n, T])
predmove[:,:,0] = numpy.reshape(initial, [n, n]) # first time step is just the prior
for k in range(1,T):
    pstate, _= hmm.prediction(initial, observations[:k])
    predmove[:,:, k] = numpy.round(numpy.reshape(pstate, [n, n]), 2) # round to make predictions stand out more

mpl.rc("font", family="serif", serif="Computer Modern", size = 24)
mpl.rc("text", usetex=True)
fs = 6

plt.figure(1) # Inference - no prediction
for t in range(T):
    plt.subplot(6, T, t+1)
    plt.imshow(noises[:,:, t], cmap="Greys", interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.title("t={0}".format(t+1),fontsize=fs)
    if t==0:
        plt.ylabel("Noises", fontsize=fs)
    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off")

    plt.subplot(6, T, t+1+T)
    plt.imshow(movements[:,:, t], cmap="Greys", interpolation="nearest")
    if t==0:
        plt.ylabel("True Location", fontsize=fs)
    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off")

    plt.subplot(6, T, t+1+2*T)
    plt.imshow(numpy.reshape(filter_[:, t], [n,n]), cmap="Greys",interpolation="nearest")
    if t==0:
        plt.ylabel("Filtering", fontsize=fs)
    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off")

    plt.subplot(6, T, t+1+3*T)
    plt.imshow(numpy.reshape(fbs[:, t], [n,n]), cmap="Greys",interpolation="nearest")
    if t==0:
        plt.ylabel("Smoothing", fontsize=fs)
    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off")

    plt.subplot(6, T, t+1+4*T)
    plt.imshow(mlmove[:,:, t], cmap="Greys",interpolation="nearest")
    if t==0:
        plt.ylabel("Viterbi", fontsize=fs)
    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off")

    plt.subplot(6, T, t+1+5*T)
    plt.imshow(predmove[:,:, t], cmap="Greys", interpolation="nearest")
    if t==0:
        plt.ylabel("Prediction", fontsize=fs)
    plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off")

#plt.show()

plt.figure(2) # Observations
fs = 34 #much larger because the pictures are bigger
plt.subplot(1,2,1)
plt.imshow(house.creaks, cmap="Greys", interpolation="nearest")
plt.title("Creaks",fontsize=fs)
plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off")

plt.subplot(1,2,2)
plt.imshow(house.bumps, cmap="Greys", interpolation="nearest")
plt.title("Bumps",fontsize=fs)
plt.tick_params(axis="both", which="both", bottom="off", top="off", left="off", right="off", labelbottom="off", labelleft="off")

plt.show()
