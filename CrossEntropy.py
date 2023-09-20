import numpy
import numpy as np


class CrossEntropy():
    def __init__(self):
        self.last_input = None
        self.last_label = None
        self.B = None
        self.Ten = None

    def forward(self, input, label):
        #avoid zero input
        input = input + 1.e-8

        #save input/label for gradient calculation
        self.last_input = input
        self.last_label = label

        self.B, self.Ten = label.shape
        output = 0
        #calculate crossentropy between input and label
        for b in range(self.B):
            for i in range(0, self.Ten):
                output = output - label[b][i]*np.log(input[b][i])

        return output/self.B

    def backward(self):
        # #calculate gradient dL/dX
        # output_gradient = np.zeros((self.B, self.Ten))
        # for b in range(self.B):
        #     for i in range(0, self.Ten):
        #         if (self.last_label[b][i] == 1):
        #             output_gradient[b][i] = -(1/self.last_input[b][i])
        #         else:
        #             output_gradient[b][i] = 0
        output_gradient = self.last_label
        return output_gradient

if __name__ == '__main__' :
    a = CrossEntropy()
    a.forward(np.array([0.2,0.2,0.6]),np.array([0,1,0]))
    print(a.forward(np.array([0.2,0.2,0.6]),np.array([0,1,0])))
    print(a.backward())
