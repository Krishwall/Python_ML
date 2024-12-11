import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# inputs=[0,2,-1,3.3,-2.7,1.1,2.2,-100]
# output=[]

# for i in inputs :
#     output.append(max(0,i))

# print(output)

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights=np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
        pass
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.biases


class Activation_ReLU:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self,inputs):
        exp_values= np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities=exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output=probabilities
X,y=spiral_data(samples=100, classes=3)

dense1=Layer_Dense(2,3) # 2 inputs - x,y coordinates, 3 neurons
activation1= Activation_ReLU()

dense2=Layer_Dense(3,3) # 3 since output of previous layer is 3
activation2 =Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
