# coding: utf-8
"""
Test of the neural network counting in binary
We give him a number x in binary and outputs x+1 and 2*x+1
"""

from LSTM import LSTM
import numpy as np


max_power = 12
learning_rate = 0.9
nn = LSTM(max_power, learning_rate, True)
nn.add_lstm_layer(max_power+10)
nn.add_simple_layer(max_power+10)
nn.add_simple_layer(max_power)
nn.graph.set_predict_stopping_condition(lambda a,b: 0)
nn.init_graph()
nn.add_progress_callback(50)

possible_numbers = np.random.permutation(2**(max_power-1))
training_ex = int(0.8*(2**(max_power-1)))
training_set = possible_numbers[: training_ex]
test_set = possible_numbers[training_ex:]

binary = [list(map(int, list(format(k, '0'+str(max_power)+'b')))) for k in range(2**max_power)]
binary_to_int = {}
for b in binary:
    binary_to_int[str(b)] = b

x_train = []
y_train = []

for x in training_set:
    ex = []
    y = []
    ex.append(binary[x])
    ex.append(binary[x+1])
    y.append(binary[x+1])
    y.append(binary[2*x+1])
    x_train.append(ex)
    y_train.append(y)

# print('x', x_train)
# print('y', y_train)

epochs = 10

nn.train(x_train, y_train, epochs)
success = 0
error = 0
for test in test_set:
    x1 = binary[test]
    print('TEST', x1)
    res = nn.predict(x1, [], 2)
    for n in range(len(res)):
        for d in range(len(res[n])):
            if res[n][d] < 0.5:
                res[n][d] = 0
            else:
                res[n][d] = 1
        if n == 0:
            key = test+1
        else:
            key = 2*test + 1
        if list(res[n]) == binary[key]:
            success += 1
            print('OK', res[n])
        else:
            error += 1
            print('ERR', res[n])

print('Error ', int(100 * error / (error+success)), '%', sep='')

# print('Test 2 counting from 1')
# test2 = nn.predict(binary[0], [], 20)
# for n in range(len(test2)):
#     test2[n] = list(test2[n])
#     for d in range(len(test2[n])):
#         if test2[n][d] < 0.5:
#             test2[n][d] = 0
#         else:
#             test2[n][d] = 1
#     # print(test2[n])
#     # print(list(map(str, list(map(int, test2[n])))))
#     dec = int(''.join(list(map(str, list(map(int, test2[n]))))), 2)
#     print(dec)