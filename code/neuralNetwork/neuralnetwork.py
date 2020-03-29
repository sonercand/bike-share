import numpy as np

class Network():
    '''
    Network contains 3 layers: input, hidden and output. 
        input -> hidden(fully connected layer) -> output(fully connected layer).
        input_layer: batch_size x number_of_features
        hidden_layer : batch_size x number_of_features X number_of_features x number_of_hidden_nodes
        activated_hidden_layer : sigmoid(hidden_layer)
        output_layer : batch_size x number_of_hidden_nodes X number_of_output_nodes(1 in this case)

    It is a regression network hense the output_layer returns a number not bounded by the range of 0 to 1(
        output does not pass through a sigmoid activation).
    
    '''
    def __init__(self, hidden_nodes, input_nodes, output_nodes, learning_rate):
        '''
        args:
            hidden_nodes : int number of hidden nodes
            output_nodes : int number of output nodes
            learning_rate: float between 0 and  1
            input_nodes  : int number of features
        '''
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.input_nodes = input_nodes
        # Random weights to start with [0,1/sqrt(input_nodes)]
        self.weights_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.sigmoid = lambda x: 1/(1+np.exp(-x))
        self.sigmoid_derivative = lambda x: x*(1-x)
        
    def print_network(self):
        input_layer = 'number of input features = {}'.format(self.input_nodes)
        hidden_layer = 'number of parameters to be trained = {}'.format(self.input_nodes*hidden_nodes)
        output_layer = 'number of parameters to be trained = {}'.format(self.hidden_nodes*self.output_nodes) 
        total_num_param = 'total number of parameters = {}'.format((self.input_nodes+self.output_nodes)*hidden_nodes)
        print(input_layer+'\n'+hidden_layer+'\n'+output_layer+'\n'+total_num_param)

    def forward_pass(self,X):
        '''
        args:
            X : numpy array feature maxtrix (batch_size x feature number)
        returns:
            output_layer: numpy array of output 
            activated_hidden_layer: numpy array of output of hidden layer after activation
        '''     
        input_layer = X
        hidden_layer = np.dot(input_layer,self.weights_hidden) 
        activated_hidden_layer = self.sigmoid(hidden_layer)
        output_layer = np.dot(activated_hidden_layer,self.weights_output)
        return output_layer, activated_hidden_layer