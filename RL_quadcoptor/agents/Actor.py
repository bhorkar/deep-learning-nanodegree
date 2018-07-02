from keras import layers 
from keras import models 
from keras import backend as K 
from keras import optimizers 
from keras import regularizers 
import copy

class Actor(object): 
     """Actor Network is a deterministic policy such that 
  
         f(states) -> action 
     """ 
 
 
     def __init__(self, input_dim: int, output_dim: int, action_low, action_high, use_batchnorm: bool = 1): 
         """Builds a network 
  
         Args: 
             input_dim (int): Dimension of `state` 
             output_dim (int): Dimension of `action`
             action_low (float): low Value of `action` 
             action_high (float): Max Value of `action` 
            use_batchnorm (bool): Use Batchnormalization 
         """ 
         self.input_dim = input_dim 
         self.output_dim = output_dim 
         self.action_high = action_high 
         self.action_low = action_low 
         self.action_range = self.action_high - self.action_low
         self.use_batchnorm = use_batchnorm 
         self.HIDDEN        = 200;
         self.L2            = 0.01
         self.CLIP_VALUE    = 40;
         self._build_network() 
         
 
 
     def _build_network(self) : 
         """Actor network architecture 
  
         Notes: 
  
             Network: 𝜇(state) -> continuous action 
  
             Loss: Policy Gradient 
  
                 mean(d_Q(s, a)/d_a * d_𝜇(s)/d_𝜃) + L2_Reg 
         """ 
         states = layers.Input(shape=(self.input_dim,), name="states") 
 
 
         # Layer1: state -> (bn) -> relu 
         net = layers.Dense(units=self.HIDDEN, 
                            kernel_regularizer=regularizers.l2(self.L2))(states) 
         if self.use_batchnorm: 
            net = layers.BatchNormalization()(net) 
            net = layers.Activation("relu")(net) 
 
 
         # Layer2: Layer1 -> (bn) -> relu 
            net = layers.Dense(units=self.HIDDEN, 
                            kernel_regularizer=regularizers.l2(self.L2))(net) 
            if self.use_batchnorm: 
               net = layers.BatchNormalization()(net) 
            net = layers.Activation("relu")(net) 
 
 
         # Layer3: Layer2 -> tanh -> actions -> actions_scaled 
            net = layers.Dense(units=self.output_dim, 
                            kernel_regularizer=regularizers.l2(self.L2))(net) 
            actions = layers.Activation("tanh")(net) 
            #actions = layers.Lambda(lambda x: x * self.action_high if x > 0 else x * self.action_low)(actions) 
            actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                    name='actions')(actions)
 
            self.model = models.Model(inputs=states, outputs=actions) 
 
 
            action_grad = layers.Input(shape=(self.output_dim,)) 
            loss = K.mean(-action_grad * actions) 
 
 
            for l2_regularizer_loss in self.model.losses: 
               loss += l2_regularizer_loss 
 
 
            optimizer = optimizers.Adam() 
            updates_op = optimizer.get_updates(params=self.model.trainable_weights, constraints = self.model.constraints,                                            
                                            loss=loss) 
 
 
            self.train_fn = K.function(inputs=[self.model.input, action_grad, K.learning_phase()], 
                                    outputs=[], 
                                    updates=updates_op) 
 
 
