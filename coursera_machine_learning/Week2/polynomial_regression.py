import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)

# create target data
x = np.arange(0, 20, 1)

# y = 1 + x**2
# y = x**2
y = np.cos(x/2)

X = x.reshape(-1, 1)
# model_w, model_b = run_gradient_descent_feng(X,y,iterations=1000,alpha=1e-2)
# X = X.reshape(-1, 1)  #X should be a 2-D Matrix
# X = np.c_[x, x**2, x**3]   #<-- added engineered feature
X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X_features = ['x','x^2','x^3']
X = zscore_normalize_features(X)

# model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)
# model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)
# model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)
model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha=1e-1)

# plt.scatter(x, y, marker='x', c='r', label='Actual Value'); plt.title("no feature engineering")
# plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
# plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
# plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")

# plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()
# plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
# plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
# plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


# fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X[:,i],y)
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("y")
# plt.show()

print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")