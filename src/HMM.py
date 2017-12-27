import numpy

class HMM:
    def __init__(self, tp, ep):
        self.tp = tp
        self.ep = ep

    def normalise(self, vec):
      # Normalise mat such that sum(vec) = 1
        return vec/sum(vec)

    def forward(self, initial, evidence):
        # Forwards algorithm for hmm
        ns = len(self.tp) #number of states
        ne = len(evidence) #number of observations
        alpha = numpy.zeros([ns, ne]) #matrix of forward probabilities

        alpha[:,0] = self.normalise(self.ep[evidence[0]]*initial)

        for ke in range(1,ne): # loop through each evidence observation
            predictions = numpy.matmul(self.tp, alpha[:,ke-1])
            alpha[:,ke] = self.normalise(self.ep[evidence[ke]]*predictions)

        return alpha

    def backward(self, evidence):
        # Backwards algorithm for hmm.
        ns = len(self.tp) #number of states
        ne = len(evidence) #number of observations
        beta = numpy.zeros([ns, ne]) #matrix of forward probabilities

        #initialise
        beta[:,-1] = 1 # for correct Bayes

        for ke in range(ne-1,0,-1): # iterate backwards over evidence, evidence at t does not matter
            beta[:, ke-1] = self.normalise(numpy.matmul(self.tp, beta[:,ke]*self.ep[evidence[ke]]))
        return beta


    def smooth(self, initial, evidence, timeLocation):
        # Forwards-Backwards algorithm. Note that it is required to split the evidence
        # accordingly.
        forwardEvidence = evidence[0:timeLocation+1]
        backwardEvidence = evidence[timeLocation:]

        alpha = self.forward(initial, forwardEvidence)[:, -1]
        beta = self.backward(backwardEvidence)[:,0]

        smoothed = self.normalise(alpha*beta)

        return smoothed

    def viterbi(self, initial, evidence):
        # The Viterbi algorithm for the most likely joint sequence of states given observations.
        ns = len(self.tp) #number of states
        ne = len(evidence) #number of observations
        mu = numpy.zeros([ns, ne]) #matrix of most likely values per state

        ## Find mu
        #initialise recursion
        mu[:,-1] = 1

        for ke in range(ne-1,0,-1): # iterate backwards over evidence

            for ks in range(ns): #iterate over each state
                mu[ks][ke-1] = self.max_viterbi(ks, evidence[ke], mu[:, ke])
            mu[:, ke-1] = self.normalise(mu[:, ke-1])

        ## Find sequence of states using mu
        mlss = numpy.zeros([ne], dtype=numpy.int)
        mlss[0] = self.arg_viterbi_init(initial, evidence[0], mu[:,0])

        for ke in range(1,ne): # find the most likely sequence of states
            mlss[ke] = self.arg_viterbi(evidence[ke], mu[:,ke], mlss[ke-1])

        return mlss

    def max_viterbi(self, state, evidence, mu_before):
        # Finds the maximum mu factor given the evidence and previous state
        ns = len(self.tp) #number of states
        vmax = 0

        for k in range(ns):
            tmax = self.tp[k][state]*self.ep[evidence][k]*mu_before[k]
            if tmax > vmax:
                vmax = tmax

        return vmax

    def arg_viterbi_init(self, initial, evidence, mu_vec):
        # Finds the most likely state given the first observation

        ns = len(self.tp) #number of states
        mls = 0 #most likely state
        smax = 0#state value maximum

        for k in range(ns):
            tmax = initial[k]*self.ep[evidence][k]*mu_vec[k]

            if tmax > smax:
                mls = k
                smax = tmax

        return mls

    def arg_viterbi(self, evidence, mu_vec, prev_mls):
        # Finds the most likely sequence of states using mu
        ns = len(self.tp) #number of states
        mls = 0 #most likely state
        smax = 0 #state value maximum
        for ks in range(ns):
            tmax = self.tp[ks][prev_mls]*self.ep[evidence][ks]*mu_vec[ks]

            if tmax > smax:
                mls = ks
                smax = tmax

        return mls

    def prediction(self, initial, evidence):
        # One step ahead hidden state and observation estimator
        ns = len(self.tp)
        ne = len(self.ep)

        ## State prediction
        pred_states = numpy.zeros(ns)
        filter_ = self.forward(initial, evidence)[:, -1]
        for ks in range(ns):
            temp = 0
            for k in range(ns):
                temp += filter_[k]*self.tp[ks][k]
            pred_states[ks] = temp

        ## Evidence Prediction
        pred_evidence = numpy.zeros(ne)
        for ke in range(ne):
            temp = 0
            for ks1 in range(ns): #h_t
                for ks2 in range(ns): #h_t+1
                    temp += filter_[ks1]*self.tp[ks2][ks1]*self.ep[ke][ks2]

            pred_evidence[ke] = temp

        return pred_states, pred_evidence

    def viterbi_dp(self, initial, evidence):
        ns = len(self.tp) #number of states
        ne = len(evidence) #number of observations
        dp = numpy.zeros([ns, ne]) #matrix of most likely values per state
        dp_ptr = numpy.zeros([ns, ne], dtype=numpy.int) #matrix of best previous state

        dp[:,0] = initial*self.ep[evidence[0]]
        v_max_end, s_max_end = -1, -1
        for ke in range(1, ne):
            for ks in range(ns):
                v_max, s_max = -1, -1
                for ks_prev in range(ns):
                    v = self.tp[ks][ks_prev]*self.ep[evidence[ke]][ks]*dp[ks_prev][ke-1]
                    if v>v_max:
                        v_max = v
                        s_max = ks_prev
                dp[ks][ke] = v_max
                dp_ptr[ks][ke] = s_max
                if ke==ne-1 and v_max>v_max_end:
                    v_max_end = v_max
                    s_max_end = ks
        
        seq = numpy.zeros([ne], dtype=numpy.int)
        ptr = s_max_end
        for ke in range(ne-1, -1, -1):
            seq[ke] = ptr
            ptr = dp_ptr[ptr][ke]
        return seq
