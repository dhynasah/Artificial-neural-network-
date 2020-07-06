
"""
Created on Mon Apr 13 16:12:15 2020

@author: Dhynasah Cakir
"""


from random import shuffle 
import numpy as np
import mnist_data_configure



class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output. np.nan_to_num is used to
        ensure numerical stability. If both a and y have a 1 then this fuction
        returns nan. and nan is converted to (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.
        """
        return (a-y)


# the list sizes contains the number of neurons in the respective layers 
#the biases and weights are initalized with random numbers 
#this assumes the first layer of neurons in an input layer
class Network(object):
    def __init__(self,sizes, cost=CrossEntropyCost):
        self.num_layers= len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights =[np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.cost = cost
        
        
    def ff(self, a):
        #return the output of the network if a is input
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
    
    #stochastic gradient descent
    #lrate is the learning rate 
    # train_data is a list of tuples x as inputs and y as desired 
    
    def SG_Descent(self, train_data, epochs, tiny_batch_size, lrate,
            lmbda = 0.0,
            evaluation_data=None):
        train_data = list(train_data)
        n= len(train_data)
        
        if evaluation_data: 
            evaluation_data = list(evaluation_data)
            n_eval = len(evaluation_data)
        for j in range(epochs):
            shuffle(train_data)
            tiny_batches = [
                train_data[i:i+tiny_batch_size]
                for i in range(0, n, tiny_batch_size)]
            for batch in tiny_batches:
                self.update_tiny_batch(
                    batch, lrate, lmbda, len(train_data))
            percent = (self.evalu(evaluation_data)/n_eval)*100
            print("{}%".format(percent))
           
       
                
    def update_tiny_batch(self, tiny_batch, lrate,lmbda, n):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in tiny_batch:
            delta_grad_b, delta_grad_w = self.backprop(x, y)
            grad_b = [nb+dnb for nb, dnb in zip(grad_b, delta_grad_b)]
            grad_w = [nw+dnw for nw, dnw in zip(grad_w, delta_grad_w)]
        self.weights = [(1-lrate*(lmbda/n))*w-(lrate/len(tiny_batch))*nw
                        for w, nw in zip(self.weights, grad_w)]
        self.biases = [b-(lrate/len(tiny_batch))*nb
                        for b, nb in zip(self.biases, grad_b)]

    def backprop(self, x, y):
        """Return a tuple ``(grad_b, grad_w)`` representing the
        gradient for the cost function C_x.  ``grad_b`` and
        ``grad_w`` are layer-by-layer lists of numpy arrays."""
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activ = x
        activs = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activ)+b
            zs.append(z)
            activ = sigmoid(z)
            activs.append(activ)
        # backward pass
        delta = (self.cost).delta(zs[-1], activs[-1], y)
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activs[-2].transpose())
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activs[-l-1].transpose())
        return (grad_b, grad_w)
    
    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.
        """
        cost = 0.0
        for x, y in data:
            a = self.ff(x)
            data_list= list(data)
            if convert: y = vect_res(y)
            cost += self.cost.fn(a, y)/len(data_list)
        cost += 0.5*(lmbda/len(data_list))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. 
        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data.
        """
        if convert:
            results = [(np.argmax(self.ff(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.ff(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    
    def evalu(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. """
        test_results = [(np.argmax(self.ff(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def vect_res(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))   


def main():

  training_data, validation_data, test_data = mnist_data_configure.data_loader()
  net = Network([784,30,10],cost =CrossEntropyCost)
  net.SG_Descent(training_data,100,10,0.10, evaluation_data=test_data)
  

  
  
if __name__ == "__main__":
    main()