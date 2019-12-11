# functions for each model
def linear_mod(a0, k, times):
    return a0*(1+k*times)

def exp_mod(a0, k, times):
    return a0*np.exp(k*times)

# residual for linear model
def resid_lin(params, areas, times):
    a0, k = params

    return areas - a0*(1+k*times)

# residual for exponential function
def resid_exp(params, areas, times):
    a0, k = params

    return areas - a0*np.exp(k*times)

# least squares regression for given residual function
def least_squares_regression(resid_fun, areas, times):
    res = scipy.optimize.least_squares(
        resid_fun, np.array([1, 0.1]), args=(areas, times), bounds=([0,-np.inf], [np.inf, np.inf])
    )

    # compute residual sum of squares from optimal params
    rss_mle = np.sum(resid_fun(res.x, areas, times)**2)

    # compute MLE for sigma
    sigma_mle = np.sqrt(rss_mle / len(areas))

    return (res.x, sigma_mle)

# let's generate area data
def gen_area_data_lin(params, times, size, rg):
    a0, k, sigma = params
    areas = linear_mod(a0, k, times)

    return np.vstack((times, rg.normal(areas, sigma))).transpose()

def gen_area_data_exp(params, times, size, rg):
    a0, k, sigma = params
    areas = exp_mod(a0, k, times)

    return np.vstack((times, rg.normal(areas, sigma))).transpose()
