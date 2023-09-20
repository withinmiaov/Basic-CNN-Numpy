import numpy
import numpy as np

class FC():
    def __init__(self, D1,D2):
        self.weights = np.random.randn(D1, D2)*np.sqrt(2/(D1))
        self.bias = np.random.randn(D2)
        # self.weights = np.array([[1,2],[3,4]])
        # self.bias = np.array([1,2])
        self.last_input = None
        self.ok_to_update = False
        self.weight_gradient = None
        self.bias_gradient = None

        self.BatchSize = None


    def forward(self,input):
        self.weight_gradient = None
        self.bias_gradient = None
        # save the input for gradient calculation
        self.last_input = input
        
        self.BatchSize = input.shape[0]
        #print('FC1_forward_input:')
        #print(input.shape)

        #fc layer forward calculation

        output = np.matmul(input, self.weights) + self.bias
        #print('FC1_forward_output:')
        #print(output.shape)
        

        return output


    def backward(self,input_gradient):
        #calculate the gradient dL/dX = dy*wT
        output_gradient = np.matmul(input_gradient, self.weights.T)

        # calculate the gradient dL/dW across miniBatch = xT*dy
        self.weight_gradient = np.matmul(self.last_input.T, input_gradient)

        # calculate the gradient dL/db across miniBatch
        self.bias_gradient = np.sum(input_gradient, axis=0)

        self.ok_to_update = True
        return output_gradient

    def weight_update(self, lr = 0.1):
        if self.ok_to_update:
            # update the weights and bias
            self.weights = self.weights - lr*self.weight_gradient / self.BatchSize
            self.bias = self.bias - lr*self.bias_gradient / self.BatchSize
            self.ok_to_update = False

if __name__ == '__main__':
    a = FC(2,2)
    print(a.weights)
    print(a.bias)
    test_input = np.zeros(2)
    for i in range(0,2):
        test_input[i] = i
    
    test_input_gradient = np.zeros(2)
    for i in range(0,2):
        test_input_gradient[i] = i

    print(test_input)
    print(test_input_gradient)
    print(a.forward(test_input))
    print(a.backward(test_input_gradient))
    print(a.weight_gradient)
    print(a.bias_gradient)
    a.weight_update()
    print(a.weights)
    print(a.bias)