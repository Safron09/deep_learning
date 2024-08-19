import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh

plt.style.use('ML_and_DL/coursera_machine_learning/Week3/deeplearning.mplstyle.md')

input_array = np.array([1, 2, 3])
exp_array = np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

input_val = 1
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """

    g = 1 / (1 + np.exp(-z))

    return g

z_tmp = np.arange(-10,11)
y = sigmoid(z_tmp)

np.set_printoptions(precision=3)
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(z_tmp, y, c='b')
ax.set_title("Sigmoid Function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax, 0)
# plt.show()

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0
plt.close('all')
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)