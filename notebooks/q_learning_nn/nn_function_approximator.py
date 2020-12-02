import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

from tensorflow import where as tf_where
import os
os.chdir('..')
import virl

def init_networks(input_layer_dim, output_layer_dim, learning_rate=0.001, nn_config=[24,24]):
    # Init the two networks
    policy_network = NNFunctionApproximatorJointKeras(learning_rate, input_layer_dim, output_layer_dim, nn_config)
    target_network = NNFunctionApproximatorJointKeras(learning_rate, input_layer_dim, output_layer_dim, nn_config)
    target_network.model.set_weights(policy_network.model.get_weights())
    return (policy_network, target_network)

def load_trained_network(filename):
    env = virl.Epidemic(stochastic=False, noisy=False)
    n_actions = env.action_space.n
    d_states = env.observation_space.shape[0]
    policy_network_new, target_network_new = init_networks(d_states, n_actions, learning_rate=0.001)
    policy_network_new.model.load_weights(filename)
    target_network_new.model.set_weights(policy_network_new.model.get_weights())
    return (policy_network_new, target_network_new)

    
    

class NNFunctionApproximatorJointKeras():
    
    def __init__(self, learning_rate, input_layer_dim, output_layer_dim, nn_config, verbose=False):        
        self.learning_rate = learning_rate 
        self.nn_config = nn_config      # determines the size of the hidden layer (if any)              
        self.input_layer_dim = input_layer_dim     
        self.output_layer_dim = output_layer_dim  
        self.verbose=verbose # Print debug information        
        self.n_layers = len(nn_config)
        self.model = self._build_model()  
                        
    def _huber_loss(self,y_true, y_pred, clip_delta=1.0):
        """
        Huber loss (for use in Keras), see https://en.wikipedia.org/wiki/Huber_loss
        The huber loss tends to provide more robust learning in RL settings where there are 
        often "outliers" before the functions has converged.
        """
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf_where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning 
        model = Sequential()
        for ilayer in self.nn_config:
            model.add(Dense(ilayer, input_dim=self.input_layer_dim, activation='relu'))        
        model.add(Dense(self.output_layer_dim, activation='linear'))
        model.compile(loss=self._huber_loss, # define a special loss function
                      optimizer=Adam(lr=self.learning_rate, clipnorm=10.)) # specify the optimiser, we clip the gradient of the norm which can make traning more robust
        return model

    def predict(self, s, a=None):              
        if a==None:            
            return self._predict_nn(s)
        else:                        
            return self._predict_nn(s)[a]
        
    def _predict_nn(self,state_hat):                          
        """
        Predict the output of the neural netwwork (note: these can be vectors)
        """                
        x = self.model.predict(state_hat)                                                    
        return x
  
    def update(self, states, td_target):
        self.model.fit(states, td_target, epochs=1, verbose=0) # take one gradient step usign Adam               
        return
