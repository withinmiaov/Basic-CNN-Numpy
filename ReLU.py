import numpy
import numpy as np


class ReLU():
    def __init__(self):
        self.x = None
        pass

    def forward(self, input):
        self.x = input
        if input.ndim == 4:
            B, H1, W1, D2 = input.shape
            # print('ReLU_forward_input:')
            # print(input.shape)
            # print(input)
            output = np.zeros((B, H1, W1, D2))

            #apply ReLU on input
            for b in range(B):
                for h1 in range(H1):
                    for w1 in range(W1):
                        for d2 in range(D2):
                            if(input[b][h1][w1][d2] >= 0):
                                output[b][h1][w1][d2] = input[b][h1][w1][d2]
                            else:
                                output[b][h1][w1][d2] = 0
            
            # print('ReLU_forward_output:')
            # print(output.shape)
            # print(output)

        elif input.ndim == 2:
            B, H = input.shape
            #print(input.shape)
            output = np.zeros((B, H))

            #apply ReLU on input
            for b in range(B):
                for h in range(H):
                    if(input[b][h] >= 0):
                        output[b][h] = input[b][h]
                    else:
                        output[b][h] = 0
        else:
            print('ERROR: ReLU_forward!')

        return output


    def backward(self,input_gradient):

        if input_gradient.ndim == 4:
            B, H1, W1, D2 = input_gradient.shape
            output_gradient = np.zeros((B, H1, W1, D2))

            #Calculate gradient dL/dX
            for b in range(B):
                for h1 in range(H1):
                    for w1 in range(W1):
                        for d2 in range(D2):
                            if(self.x[b][h1][w1][d2] >= 0):
                                output_gradient[b][h1][w1][d2] = input_gradient[b][h1][w1][d2]
                            else:
                                output_gradient[b][h1][w1][d2] = 0
        elif input_gradient.ndim == 2:
            B, H = input_gradient.shape
            #print(input.shape)
            output_gradient = np.zeros((B, H))

            #apply ReLU on input
            for b in range(B):
                for h in range(H):
                    if(self.x[b][h] >= 0):
                        output_gradient[b][h] = input_gradient[b][h]
                    else:
                        output_gradient[b][h] = 0
        else:
            print('ERROR: ReLU_backward!')

        return output_gradient

if __name__ == '__main__':
    a = ReLU()
    input = np.zeros((2, 6, 6, 8))
    a.forward(input)
    #print(a.backward(-10))
