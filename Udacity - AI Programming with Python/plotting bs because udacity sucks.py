#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:24:36 2018

@author: Jake
"""

import numpy as np
import matplotlib.pyplot as plt

ax = plt.axes()
ax.plot(0,0,'bo')

plt.ylim((-2,2))
plt.xlim((-2,2))

plt.plot([1,0],[0,1], 'k-', lw = 2)
plt.plot([0,0], 'bo')
plt.plot([0,1], 'ro')
plt.plot([1,0], 'ro')
plt.plot([1,1], 'ro')
plt.grid(b = True, which = 'major')

plt.show()

print('\nProblem 2\n')

x_weight = 0.0 # m
y_weight = -1.0 # n
bias = 0.5

y_points = []
x_points = np.arange(0, 1.1, 0.1)

for i in np.nditer(x_points):
    y_points.append(x_weight * i + bias)
    y = x_weight * i + bias
    print('x:', round(float(i),2), ' | y:', round(float(y),2))

plt.plot(x_points, y_points, 'k-')
plt.plot([0,0], 'bo')
plt.plot([0,1], 'ro')
plt.plot([1,0], 'ro')
plt.plot([1,1], 'ro')
plt.grid(b = True, which = 'major')
plt.show()

##########

# TODO: Set weight1, weight2, and bias
weight1 = 0.0
weight2 = -1.0
bias = 0.5


# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [True, False, True, False]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))