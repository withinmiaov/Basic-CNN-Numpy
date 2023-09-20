from image_loader import load_mnist
from Conv import Conv
from FC import FC
from Pooling import  AvgPool
from ReLU import ReLU
from Softmax import Softmax
from CrossEntropy import CrossEntropy
import numpy as np


#load mnist dataset
train_data,train_label,test_data, test_label = load_mnist()

class ConvNet():
    def __init__(self):
        self.conv_1 = Conv(5, 5, 1, 8)
        self.relu_1 = ReLU()
        self.pooling_1 = AvgPool(2)
        self.fc_1 = FC(1152, 128)
        self.relu_2 = ReLU()
        self.fc_2 = FC(128, 10)
        self.softmax_1 = Softmax()
        self.L = CrossEntropy()

        self.pooling_output_d1 = None
        self.pooling_output_d2 = None
        self.pooling_output_d3 = None
        self.pooling_output_d4 = None

    def forward(self, input, label):
        temp_output_1 = self.conv_1.forward(input)
        temp_output_2 = self.relu_1.forward(temp_output_1)
        temp_output_3 = self.pooling_1.forward(temp_output_2)
        self.pooling_output_d1, self.pooling_output_d2, self.pooling_output_d3, self.pooling_output_d4 = temp_output_3.shape
        temp_output_3 =np.reshape(temp_output_3,(temp_output_3.shape[0], -1))

        temp_output_4 = self.fc_1.forward(temp_output_3)
        temp_output_5 = self.relu_2.forward(temp_output_4)
        temp_output_6 = self.fc_2.forward(temp_output_5)
        temp_output_7 = self.softmax_1.forward(temp_output_6)
        return self.L.forward(temp_output_7, label)
    
    def backward(self):
        temp_output_1 = self.L.backward()
        temp_output_2 = self.softmax_1.backward(temp_output_1)
        temp_output_3 = self.fc_2.backward(temp_output_2)
        temp_output_4 = self.relu_2.backward(temp_output_3)
        temp_output_5 = self.fc_1.backward(temp_output_4)

        temp_output_5 =np.reshape(temp_output_5,(self.pooling_output_d1, self.pooling_output_d2, self.pooling_output_d3, self.pooling_output_d4))

        temp_output_6 = self.pooling_1.backward(temp_output_5)
        temp_output_7 = self.relu_1.backward(temp_output_6)
        self.conv_1.backward(temp_output_7)

    def update(self, lr):
        #print(self.conv_1.weight_gradient)
        self.conv_1.weight_update(lr)
        self.fc_1.weight_update(lr)
        self.fc_2.weight_update(lr)


#instantiate the network layers
myConvNet = ConvNet()

print('Successful: Create')

#define the training params
batch_size = 256
N_iter = train_data.shape[0]//batch_size
lr = 0.02


#train the network
for epoch in range(5):
    for i in range(N_iter):
        input = train_data[i*batch_size:(i+1)*batch_size,:,:,:]
        # print('input_shape')
        # print(input.shape)
        label = np.eye(10)[train_label[i*batch_size:(i+1)*batch_size]]
        # print('label_shape')
        # print(label.shape)

        # forwardpropagation
        L = myConvNet.forward(input, label)

        #if epoch % 1 == 0 and i == 0:
        if epoch % 1 == 0:
            print("epoch:", epoch, \
                  " iteration:", i, \
                  ' Loss:', L, \
                  ' Accuracy:', ((batch_size - np.count_nonzero(np.argmax(label, axis=1) - np.argmax(myConvNet.L.last_input, axis=1)))/batch_size)*100, '%')

        # backpropagation
        myConvNet.backward()

        # weight update
        myConvNet.update(lr)


#test the network
N = 0
n= 0
N_iter = test_data.shape[0]//batch_size

for i in range(N_iter):
    input = test_data[i*batch_size:(i+1)*batch_size,:,:,:]
    label = np.eye(10)[test_label[i*batch_size:(i+1)*batch_size]]

    #inference
    L = myConvNet.forward(input, label)

    #get the label
    N += batch_size
    n += batch_size - np.count_nonzero(np.argmax(label, axis=1) - np.argmax(myConvNet.L.last_input, axis=1))

# calculate the accuracy
print("final accuracy for test is: ", n/N*100,"%" )
