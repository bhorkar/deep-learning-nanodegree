from keras import layers 
from keras import models 
from keras import backend as K 
from keras import optimizers 
from keras import regularizers 
import copy
import numpy as np

class Critic(object): 
     """Critic Network is basically a function such that 
  
         f(state, action) -> value (= Q_value) 
     """ 
 
 
     def __init__(self, input_dim: int, output_dim: int, use_batchnorm: bool = 1): 
         """Builds a network at initialization 
  
         Args: 
             input_dim (int): Dimension of `state` 
             output_dim (int): Dimension of `action` 
             use_batchnorm (bool): Use BatchNormalization if `True` 
         """ 
         self.input_dim = input_dim 
         self.output_dim = output_dim 
         self.use_batchnorm = use_batchnorm 
         self.HIDDEN        = 200;
         self.L2            = 0.01
         self.CLIP_VALUE    = 40;
         self._build_network() 
 
     def _build_network(self) : 
         """Critic Network Architecture 
  
         (1) [states] -> fc -> (bn) -> relu -> fc 
         (2) [actions] -> fc 
         (3) Merge[(1) + (2)] -> (bn) -> relu 
         (4) [(3)] -> fc (= Q_pred) 
  
         Notes: 
             `Q_grad` is `d_Q_pred/d_action` required for `ActorNetwork` 
         """ 
         states = layers.Input(shape=(self.input_dim,), name="states") 
         actions = layers.Input(shape=(self.output_dim,), name="actions") 
 
         # Layer 1: states -> fc -> (bn) -> relu 
         net = layers.Dense(units=self.HIDDEN, 
                            kernel_regularizer=regularizers.l2(self.L2))(states) 
         if self.use_batchnorm: 
             net = layers.BatchNormalization()(net) 
         net = layers.Activation("relu")(net) 
 
         # Layer 2: 
         # Merge[Layer1 -> fc, actions -> fc] -> (bn) -> relu 
         states_out = layers.Dense(units=self.HIDDEN, 
                                   kernel_regularizer=regularizers.l2(self.L2))(net) 
         actions_out = layers.Dense(units=self.HIDDEN, 
                                    kernel_regularizer=regularizers.l2(self.L2))(actions) 
         net = layers.Add()([states_out, actions_out]) 
         if self.use_batchnorm: 
             net = layers.BatchNormalization()(net) 
         net = layers.Activation("relu")(net) 
 
 
         # Layer 3: Layer2 -> fc 
         Q_pred = layers.Dense(units=1, 
                               kernel_regularizer=regularizers.l2(self.L2))(net) 
 
         Q_grad = K.gradients(Q_pred, actions) 
         self.model = models.Model(inputs=[states, actions], outputs=Q_pred) 
 
 
         optimizer = optimizers.Adam() 
         self.model.compile(optimizer=optimizer, loss="mse") 
         self.get_Q_grad = K.function(inputs=[*self.model.input, K.learning_phase()], 
                                      outputs=Q_grad) 
 
 
     def get_action_gradients(self,inputs):
                return np.reshape(self.get_Q_grad(inputs), (-1, self.output_dim)) 
    