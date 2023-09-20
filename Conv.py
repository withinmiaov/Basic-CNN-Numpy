import numpy
import numpy as np

class Conv():
    def __init__(self, k1,k2,D1,D2):
        self.weights = np.random.randn(k1, k2, D1, D2)*np.sqrt(2/(k1*k2*D1))
        self.bias = np.random.randn(D2)
        self.last_input = None
        self.ok_to_update = False
        self.weight_gradient = None
        self.bias_gradient = None

        self.k1 = k1
        self.k2 = k2
        self.D1 = D1
        self.D2 = D2

        self.BatchSize = None


    def forward(self,input):
        self.weight_gradient = None
        self.bias_gradient = None

        #save the input for gradient calculation
        self.last_input = input
        B, H, W, D1 = input.shape
        self.BatchSize = B
        k1, k2, D1, D2 = self.weights.shape
        W1 = W - self.k2 + 1
        H1 = H - self.k1 + 1 
        output = np.zeros((B, H1, W1, D2))

        for n in range(0, B):
            for d2 in range(0, D2):
                for h1 in range(0, H1):
                    for w1 in range(0, W1):
                        output[n][h1][w1][d2] = 0
                        for i in range(0, k1):
                            for j in range(0, k2):
                                for d1 in range(0, D1):
                                    h = h1 + i
                                    w = w1 + j
                                    output[n][h1][w1][d2] += input[n][h][w][d1]*self.weights[i][j][d1][d2]
        for n in range(0, B):
            for d2 in range(0, D2):
                for h1 in range(0, H1):
                    for w1 in range(0, W1):
                        output[n][h1][w1][d2] += self.bias[d2]

        #print('conv_forward_output:')
        # print(output.shape)
        #print(output)

        return output


    def backward(self,input_gradient):
        B, H1, W1, D2 = input_gradient.shape
        B, H,  W, D1 = self.last_input.shape
        k1, k2, D1, D2 = self.weights.shape

        #calculate the gradient dL/dX = (W * dL/dy)
        output_gradient = np.zeros((B, H, W, D1))

        for n in range(0, B):
            for d2 in range(0, D2):
                for h1 in range(0, H1):
                    for w1 in range(0, W1):
                        #
                        for i in range(0, k1):
                            for j in range(0, k2):
                                for d1 in range(0, D1):
                                    h = h1 + i
                                    w = w1 + j
                                    output_gradient[n][h][w][d1] += self.weights[i][j][d1][d2] * input_gradient[n][h1][w1][d2]

        # print('output_gradient:')
        # print(output_gradient.shape)
        # calculate the gradient dL/dW across miniBatch  dL/dW = (dL/dy * x) 
        self.weight_gradient = np.zeros((k1, k2, D1, D2))
        for n in range(0, B):
            for d2 in range(0, D2):
                for h1 in range(0, H1):
                    for w1 in range(0, W1):
                        #
                        for i in range(0, k1):
                            for j in range(0, k2):
                                for d1 in range(0, D1):
                                    h = h1 + i
                                    w = w1 + j
                                    self.weight_gradient[i][j][d1][d2] += input_gradient[n][h1][w1][d2]*self.last_input[n][h][w][d1]

        # print('weight_gradient:')
        # print(self.weight_gradient.shape)

        # calculate the gradient  dL/db across miniBatch
        self.bias_gradient = np.zeros(D2)
        for n in range(0, B):
            for d2 in range(0, D2):
                for h1 in range(0, H1):
                    for w1 in range(0, W1):
                        self.bias_gradient[d2] += input_gradient[n][h1][w1][d2]
        # print('bias_gradient:')
        # print(self.bias_gradient.shape)


        self.ok_to_update = True
        return output_gradient

    def weight_update(self, lr = 0.1):
        if self.ok_to_update:
            #update the weights and bias
            #Directly add
            self.weights = self.weights - lr * self.weight_gradient / self.BatchSize
            self.bias = self.bias - lr * self.bias_gradient/ self.BatchSize
            self.ok_to_update = False
        else:
            print('ERROR:weight_update---ok_to_update')


if __name__ == '__main__':
    conv = Conv(5, 5, 1, 8)
    input = np.random.randn(2, 6, 6, 1)
    #print(input)
    output = conv.forward(input)
    conv.backward(output)
    #weights = np.random.randn(5, 5, 1, 8)*np.sqrt(2/(5*5*1))
    #print(type(weights))
    #print(weights)