import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from linear_models import RidgeRegScratch

X, y = make_x_y(deg=9)


def make_x_y(deg=2):
    X = np.array([*range(-100, 100)]).reshape(-1, 1) / 100

    # rands = np.random.uniform(low=-0.05, high=0.05, size=X.shape[0]).reshape(-1,1)
    # X += rands

    poly_adder = PolynomialFeatures(degree=deg)
    X = poly_adder.fit_transform(X)

    thetas = np.array(np.random.randn(deg+1, 1)).reshape(-1, 1)
    # return thetas

    y = X.dot(thetas)
    y += np.random.normal(loc=0, scale=.1, size=(len(y), 1))
    return X, y


def plot_alphas(X, y, alphas=[0.0001, 10, 1000000000000], show_degree=1):
    fig, (ax, ax1, ax2) = plt.subplots(1, len(alphas), figsize=(20, 10))
    for alpha_, ax_ in zip(alphas, [ax, ax1, ax2]):
        model = RidgeRegScratch(alpha=alpha_)
        model.fit(X, y)
        # uncomment the below line to show the predicted coefficients for each iteration of alpha
        # note, that the coefficient theta_0 remains very similar, while all other coefficients
        # get progressively smaller as alpha grows larger

        # print(f'thetas for alpha = {alpha_}: {model.thetas.T}')
        predictions = model.predict(X)
        ax_.scatter(X[:, show_degree], y)
        ax_.plot(X[:, show_degree], predictions, color='red')
        ax_.set_title(f'Alpha = {alpha_}')
        ax_.set_ylabel('y')
        ax_.set_xlabel(f'X degree {show_degree}')
    fig.suptitle(
        'Ridge Regression model fits for different tuning parameters alpha', size=20)
    fig.show()
    return fig


fig_alphas = plot_alphas(X, y, show_degree=1)
