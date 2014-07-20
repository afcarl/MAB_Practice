import scipy.stats as stats
import numpy as np
from pymc import rbeta

rand = np.random.rand
beta = stats.beta

########################
# Bandit's Structure

class GeneralBanditStrat( object ):	
    """
    Implements a online, learning strategy to solve
    the Multi-Armed Bandit problem.
    
    parameters:
        bandits: a Bandit class with .pull method
		choice_function: accepts a self argument (which gives access to all the variables), and 
						returns and int between 0 and n-1
    methods:
        sample_bandits(n): sample and train on n pulls.

    attributes:
        N: the cumulative number of samples
        choices: the historical choices as a (N,) array
        bb_score: the historical score as a (N,) array
    """
    
    def __init__(self, bandits, choice_function):
        
        self.bandits = bandits
        n_bandits = len( self.bandits )
        self.wins = np.zeros( n_bandits )
        self.trials = np.zeros( n_bandits )
        self.N = 0
        self.choices = []
        self.score = []
        self.choice_function = choice_function

    def sample_bandits( self, n=1 ):
        ##INPUT: n: number of pulls/trials
        ##OUTPUT: updated array of scores update array of choices

        score = np.zeros( n ) #array: win(1) or loss(0) by trial
        choices = np.zeros( n ) #array: machine chosen by trial
        
        for k in range(n):
            #sample from the bandits's priors, and select the largest sample
            choice = self.choice_function( self ) #choice=index 
            
            #sample the chosen bandit
            result = self.bandits.pull( choice )
            
            #update priors and score
            self.wins[ choice ] += result
            self.trials[ choice ] += 1
            score[ k ] = result 
            self.N += 1
            choices[ k ] = choice
            
        self.score = np.r_[ self.score, score ] #array: win(1) or loss(0) by trial
        self.choices = np.r_[ self.choices, choices ] #array: machine chosen by trial
        return 

            
    def max_mean(self):
        """pick the bandit with the current best observed proportion of winning """
        return np.argmax( self.wins / ( self.trials +1 ) )

    
class Bandits(object):
    """
    This class represents N bandits machines.

    parameters:
        p_array: a (n,) Numpy array of probabilities >0, <1.

    methods:
        pull( i ): return the results, 0 or 1, of pulling 
                   the ith bandit.
    """
    def __init__(self, p_array):
        self.p = p_array
        self.optimal = np.argmax(p_array)
        
    def pull( self, i ):
        #i is which arm to pull
        return rand() < self.p[i]
    
    def __len__(self):
        return len(self.p)

 ######################
 # Bandit algorithms / strategies       
def random_choice( self):
    return np.random.randint( 0, len( self.wins ) ) # len is num of machines
    
def bayesian_bandit(self):
    alpha = 1 + self.wins
    b = 1 + self.trials - self.wins # beta var and this is your losses
    return np.argmax( rbeta( alpha, b) )

def epsilon_greedy(self):
    epsilon = 0.1
    if rand() < epsilon:
        return np.random.randint( 0, len( self.wins ) ) # len is num of machines
    else:
        return self.max_mean()
           
def ucb1(self):
    if self.N < 4:
        return self.N
    else:
        means = self.wins / ( self.trials +1 )
        UCBs = means + np.sqrt(2*np.log(self.N)/(self.trials))
        return np.argmax(UCBs)

def softmax(self):
    pass

########################
# Performance assessment

def regret( probabilities, choices ):
    w_opt = probabilities.max()
    return ( w_opt - probabilities[choices.astype(int)] ).cumsum()

def optimal( probabilities, choices ):
    # add up the total show of opitmal in choices divide by len of choices
    print float(list(choices).count(bandits.optimal)) / len(choices))
    return float(list(choices).count(bandits.optimal) / len(choices))