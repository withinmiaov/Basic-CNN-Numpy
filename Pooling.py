import numpy
import numpy as np


class AvgPool():
    def __init__(self, k):
        self.k = k

    def forward(self,input):
        # Dimensions of the input array
        B, H, W, D1  = input.shape

        # Dimensions of the output array
        OH = H // self.k
        OW = W // self.k

        #Apply average pool of input
        output = np.zeros((B,OH,OW,D1))

        # print('pooling_forward_input:')
        # print(input.shape)
        # print(input)

            
        for i in range(0,B):
            for j in range(0,D1):
                for x in range(0,OH):
                    for y in range(0,OW):
                        output[i][x][y][j] = (input[i][x*2][y*2][j] + input[i][x*2+1][y*2][j] + input[i][x*2][y*2+1][j] + input[i][x*2+1][y*2+1][j])/4
        
        #print('pooling_forward_output:')
        #print(output.shape)
        #print(output)
        return output
    
    def backward(self,input_gradient):
        # Dimensions of the input array
        B, H, W, D1  = input_gradient.shape

        # Dimensions of the output array
        OH = H * self.k
        OW = W * self.k

        output_gradient = np.zeros((B,OH,OW,D1))
        #calculate gradient dL/dX
        for i in range(0,B):
            for j in range(0,D1):
                for x in range(0,OH):
                    for y in range(0,OW):
                        a = x // 2
                        b = y // 2
                        output_gradient[i][x][y][j] = input_gradient[i][a][b][j]/4

        return output_gradient

if __name__ == '__main__':
    test_input = np.zeros((1,4,4,1))
    for i in range(0,4):
            for j in range(0,4):
                test_input[0][i][j][0] = i + j
    
    print(test_input)
    a = AvgPool(2)
    print(a.forward(test_input))
    print(a.backward(test_input))