import numpy
import numpy as np

class Softmax():
    def __init__(self):
        self.out = None
        self.B = None
        self.Ten = None

    def forward(self,input):
        self.B, self.Ten = input.shape
        #print(input.shape)
        #print(input)
        output = np.zeros((self.B, self.Ten))
        #calculate the softmax of input
        for b in range(self.B):
            exp_sum = 0
            for i in range(self.Ten):
                exp_sum += np.exp(input[b][i])
            # print('exp_sum=', exp_sum)
            for i in range(self.Ten):
                output[b][i] = np.exp(input[b][i]) / exp_sum
        
        #save the cross entropy for gradient calculation
        self.out = output
        # print('softmax_forward')
        # print(output)

        return output

    def backward(self,input_gradient):
        output_gradient = np.zeros((self.B, self.Ten))

        # ##calculate gradient dL/dX
        # for b in range(self.B):
        #     for i in range(self.Ten):
        #         for j in range(self.Ten):
        #             if i == j:
        #                 output_gradient[b][i] += input_gradient[b][j] - input_gradient[b][j] * input_gradient[b][j]
        #             else:
        #                 output_gradient[b][i] += - input_gradient[b][i] * input_gradient[b][j]
        #         output_gradient[b][i] = output_gradient[b][i]/10

        ##calculate gradient dL/dX
        # print('softmax_backward')
        # print(output_gradient)
        for b in range(self.B):
            for i in range(self.Ten):
                output_gradient[b][i] = self.out[b][i] - input_gradient[b][i]
        
        # print(output_gradient)
        return output_gradient

if __name__ == '__main__':
    a = Softmax()
    a.forward(10)
    print(a.backward(10))
