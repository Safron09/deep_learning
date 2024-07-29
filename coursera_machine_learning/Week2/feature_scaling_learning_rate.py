import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import  load_house_data, run_gradient_descent
from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc

np.set_printoptions(precision=2)
plt.style.use('ML_and_DL/coursera_machine_learning/Week2/deeplearning.mplstyle.md')

X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price(1000's)")
plt.show()

# _, _, hist = run_gradient_descent(X_train, y_train, 10, alpha=9.9e-7)
# plot_cost_i_w(X_train, y_train, hist)
# _, _, hist = run_gradient_descent(X_train, y_train, 10, alpha=9e-7)
# plot_cost_i_w(X_train, y_train, hist)
_, _, hist = run_gradient_descent(X_train, y_train, 10, alpha=1e-7)
plot_cost_i_w(X_train, y_train, hist)

def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu = np.mean(X, axis=0)  # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std(X, asix=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X-mu)/sigma
    X_mean = (X_train-mu)
    return(X_norm, mu, sigma)

    fig,ax=plt.subplots(1, 3, figsize=(12, 3))
    ax[0].scatter(X_train[:, 0],X_train[:, 3])
    ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
    ax[0].set_title("unnormalized")
    ax[0].axis("equal")

    ax[1].scatter(X_mean[:, 0], X_mean[:, 3])
    ax[1].set_xlabel(X_features[0], ax[0].set_ylabel(X_features[3]))




