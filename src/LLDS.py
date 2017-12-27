import numpy

class llds:
  # Assume zero mean transition and emission defs.
  # The linear latent dynamical system should have the
  # state space form:
  # x(t+1) = A*x(t) + Bu(t) + Q
  # y(t+1) = C*x(t+1) + R (there could be Du(t) term here but we assume the inputs don't affect
  # the measurements directly.)
  # I assume the simplest self I will deal with has matrix A, B and float C therefore
  # the slightly parametric type. Also that specific simple case has only one input.
  # this is to avoid ugly notation later.
    def __init__(self, A, B, C, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q # Process Noise
        self.R = R # Measurement Noise VARIANCE


    def step(self, xprev, uprev):
        # Controlled, move multivariate self one time step forward.
        xnow = numpy.matmul(self.A, xprev) + self.B*uprev
        ynow = numpy.matmul(self.C, xnow)

        return xnow,  ynow
        
    
    def init_filter(self, initmean, initvar, ynow):
        # Initialise the filter. No prediction step, only a measurement update step.
        updatedMean , updatedVar  = self.step_update(initmean, initvar, ynow)
        return updatedMean, updatedVar
        

    def step_filter(self, prevmean, prevvar, uprev, ynow):
        # Return the posterior over the current state given the observation and previous
        # filter result.
        pmean , pvar  = self.step_predict(prevmean, prevvar, uprev)
        updatedMean , updatedVar  = self.step_update(pmean, pvar, ynow)
        return updatedMean, updatedVar
        

    def step_predict(self, xprev, varprev, uprev):
        # Return the one step ahead predicted mean and covariance.
        pmean = numpy.matmul(self.A, xprev) + self.B*uprev
        pvar =  self.Q + numpy.matmul(numpy.matmul(self.A, varprev), numpy.transpose(self.A))
        return pmean, pvar
        

    def step_update(self, pmean, pvar, ymeas):
        # Return the one step ahead measurement updated mean and covar.
        temp = numpy.matmul(numpy.matmul(self.C, pvar), numpy.transpose(self.C))
        inverse = numpy.linalg.inv(temp + self.R)
        temp2 = numpy.matmul(pvar, numpy.transpose(self.C))
        if temp2.ndim==1:
            temp2 = numpy.transpose([temp2])
        
        kalmanGain = numpy.matmul(temp2, inverse)
        ypred = numpy.matmul(self.C, pmean) #predicted measurement
        if ymeas.ndim==0:
            updatedMean = pmean + numpy.transpose(kalmanGain*(ymeas - ypred))
        else:
            updatedMean = pmean + numpy.transpose(numpy.matmul(kalmanGain, (ymeas - ypred)))
        rows, cols = numpy.shape(pvar)
        updatedVar = numpy.matmul((numpy.eye(rows) - kalmanGain*self.C), pvar)
        return updatedMean, updatedVar
        

    def smooth(self, kmeans, kcovars, us):
        # Returns the smoothed means and covariances
        # Note, this is only for matrix entries!
        rows, cols = numpy.shape(kmeans)
        smoothedmeans = numpy.zeros([rows, cols])
        smoothedvars = numpy.zeros([rows, rows, cols])
        smoothedmeans[:, -1] = kmeans[:, -1]
        smoothedvars[:, :, -1] = kcovars[:, :, -1]

        for t in range(cols-2, 1, -1):
            Pt = self.A*kcovars[:, :, t]*numpy.transpose(self.A) + self.Q
            Jt = kcovars[:, :, t]*numpy.transpose(self.A)*numpy.linalg.inv(Pt)
            smoothedmeans[:, t] = kmeans[:, t] + Jt*(smoothedmeans[:, t+1] - self.A*kmeans[:, t] - self.B*us[:, t-1])
            smoothedvars[:,:, t] = kcovars[:,:, t] + Jt*(smoothedvars[:,:, t+1] - Pt)*numpy.transpose(Jt)
            

        Pt = self.A*kcovars[:, :, 0]*numpy.transpose(self.A) + self.Q
        Jt = kcovars[:, :, 0]*numpy.transpose(self.A)*numpy.linalg.inv(Pt)
        smoothedmeans[:, 0] = kmeans[:, 0] + Jt*(smoothedmeans[:, 1] - self.A*kmeans[:, 0]) # no control for the prior
        smoothedvars[:,:, 0] = kcovars[:,:, 0] + Jt*(smoothedvars[:,:, 1] - Pt)*numpy.transpose(Jt)


        return smoothedmeans, smoothedvars
        

    def predict_visible(self, kmean, kcovar, us):
        # Predict the visible states n steps into the future given the controller action.
        # Note: us[t] predicts xs[t+1]

        rows = len(kmean)
        n = len(us)

        predicted_means, predicted_covars = predict_hidden(kmean, kcovar, us)

        rows = len(self.R) # actually just the standard deviation
        predicted_vis_means = numpy.zeros(rows, n)
        predicted_vis_covars = numpy.zeros(rows, n)

        for k in range(n): # convert the hidden state to the observed state
            predicted_vis_means[:, k] = self.C*predicted_means[:,k]

            predicted_vis_covars[:, k] = self.R + self.C*predicted_covars[:, :, k]*numpy.transpose(self.C)
        

        return predicted_vis_means, predicted_vis_covars
        

    def predict_hidden(self, kmean, kcovar, us):
        # Predict the hidden states n steps into the future given the controller action.
        # Note: us[t] predicts xs[t+1]

        rows = len(kmean)
        n  = len(us)
        predicted_means = numpy.zeros([rows, n])
        predicted_covars = numpy.zeros([rows, rows, n])

        predicted_means[:, 0] = self.A*kmean + self.B*us[0]
        predicted_covars[:, :, 0] = self.Q + self.A*kcovar*numpy.transpose(self.A)

        for k in range(1,n): #cast the state forward
            predicted_means[:, k], predicted_covars[:, :, k] = step_predict(predicted_means[:,k-1], predicted_covars[:, :, k-1],us[k])
        

        return predicted_means, predicted_covars
        


