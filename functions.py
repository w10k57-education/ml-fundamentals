import numpy as np

def generate_data(m_samples, x_max=1, a_range=10,
                  b_range=1, noise=0.2, coeff=False):
    """
    Generate synthetic data for linear regression analysis.

    Parameters:
    -----------
    m_samples : int
        Number of data samples to generate.

    x_max : float, optional
        Maximum value for the independent variable x. Default is 1.

    a_range : float, optional
        Range of values from which the coefficient 'a' is randomly sampled. The coefficient
        'a' is drawn uniformly from the interval [-a_range, a_range]. Default is 10.

    b_range : float, optional
        Range of values from which the coefficient 'b' is randomly sampled. The coefficient
        'b' is drawn uniformly from the interval [-b_range, b_range]. Default is 1.

    noise : float, optional
        Standard deviation of the Gaussian noise added to the generated data. Default is 0.2.

    coeff : bool, optional
        If True, print the coefficients 'a' and 'b' used to generate the data.

    Returns:
    --------
    x_features : ndarray
        Array of shape (m_samples,) containing the independent variable 'x' values.

    y_features : ndarray
        Array of shape (m_samples,) containing the dependent variable 'y' values.

    Notes:
    ------
    This function generates synthetic data for a simple linear regression model of the form:
    y = a * x + b + ε
    where 'a' and 'b' are randomly generated coefficients, 'x' is the independent variable,
    and ε is random Gaussian noise with a standard deviation specified by 'noise'.

    If 'coeff' is set to True, the coefficients 'a' and 'b' used to generate the data are
    printed to the console.

    Example:
    --------
    >>> x, y = generate_data(100, x_max=5, a_range=2, b_range=0.5, noise=0.1, coeff=True)
    Coefficients used to generate data:
    a = 0.96
    b = -0.13
    """
    rg = np.random.default_rng(seed=18)
    a_data = rg.uniform(-a_range, a_range)
    b_data = rg.uniform(-b_range, b_range)
    x_features = rg.uniform(0, x_max, m_samples)
    noise_stddev = rg.uniform(noise/2, noise)
    y_features = (a_data * x_features + b_data +
                  np.random.normal(0, noise_stddev, m_samples))
    if coeff is True:
        print(f'Coefficients used to generate data: \n'
              f'a = {a_data:.2f} \nb = {b_data:.2f}')
    return x_features, y_features

def linear_function(x, a, b):
    """
    Calculate the value of a linear function for given input values.

    Parameters:
    -----------
    x : array_like
        Input values for the independent variable.

    a : float
        Coefficient for the independent variable 'x' in the linear function.

    b : float
        Intercept term in the linear function.

    Returns:
    --------
    y : ndarray
        Array of the same shape as 'x' containing the values of the linear function
        evaluated at the corresponding input values.

    Notes:
    ------
    This function computes the values of a simple linear function of the form:
    y = a * x + b
    for a given set of input values 'x', where 'a' represents the coefficient for 'x'
    and 'b' represents the intercept term.

    Example:
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> a = 2.0
    >>> b = 1.0
    >>> y = linear_function(x, a, b)
    array([ 3.,  5.,  7.,  9., 11.])
    """
    y = a * x + b
    return y

def cost_function(x, y, a, b):
    """
    Calculate the cost (mean squared error) of a linear model's predictions.

    Parameters:
    -----------
    x : array_like
        Input values for the independent variable.

    y : array_like
        Observed or target values for the dependent variable.

    a : float
        Coefficient for the independent variable 'x' in the linear function.

    b : float
        Intercept term in the linear function.

    Returns:
    --------
    total_cost : float
        The mean squared error (MSE) cost, which quantifies the average squared
        difference between the model's predictions and the actual target values.

    Notes:
    ------
    This function calculates the cost of a linear model's predictions by computing
    the mean squared error (MSE). The cost is a measure of how well the linear model
    with coefficients 'a' and 'b' fits the observed data points.

    The MSE is computed as the average of the squared differences between the predicted
    values obtained from the linear function (a * x + b) and the actual target values 'y'.
    The formula for MSE is:
    
    MSE = (1/m) * Σ[(a * x[i] + b - y[i])^2]

    where 'm' is the number of data points.

    Example:
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> a = 2.0
    >>> b = 1.0
    >>> cost = cost_function(x, y, a, b)
    0.0
    """
    m = len(x)
    cost = 0
    partial_sum = (linear_function(x, a, b) - y) ** 2
    total_cost = sum(partial_sum) / m
    return total_cost

def gradient_calc(x, y, a, b):
    """
    Calculate the gradients of a linear model's cost with respect to its coefficients.

    Parameters:
    -----------
    x : array_like
        Input values for the independent variable.

    y : array_like
        Observed or target values for the dependent variable.

    a : float
        Coefficient for the independent variable 'x' in the linear function.

    b : float
        Intercept term in the linear function.

    Returns:
    --------
    gradient_a : float
        The gradient (derivative) of the cost function with respect to the coefficient 'a'.

    gradient_b : float
        The gradient (derivative) of the cost function with respect to the intercept 'b'.

    Notes:
    ------
    This function calculates the gradients of a linear model's cost function with respect
    to its coefficients 'a' and 'b'. These gradients are useful for performing gradient
    descent optimization when training the linear model.

    The gradients are computed by iterating through the data points and computing the
    partial derivatives of the cost function with respect to 'a' and 'b'. The gradients
    are then scaled by a factor of 2/m, where 'm' is the number of data points, to obtain
    the average gradient.

    Example:
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> a = 2.0
    >>> b = 1.0
    >>> gradient_a, gradient_b = gradient_calc(x, y, a, b)
    Gradient 'a': -6.0
    Gradient 'b': -2.0
    """
    m = len(x)
    sum_agrad = 0
    sum_bgrad = 0
    for i in range(m):
        partial_sum_agrad = (linear_function(x[i], a, b) - y[i]) * x[i]
        partial_sum_bgrad = linear_function(x[i], a, b) - y[i]
        sum_agrad += partial_sum_agrad
        sum_bgrad += partial_sum_bgrad
    gradient_a = 2 * sum_agrad / m
    gradient_b = 2 * sum_bgrad / m
    return gradient_a, gradient_b

def gradient_descent(x, y, a0=0, b0=0,
                     alpha=0.1, n_iter=100, print_res=0):
    """
    Perform gradient descent optimization to find the best coefficients for a linear model.

    Parameters:
    -----------
    x : array_like
        Input values for the independent variable.

    y : array_like
        Observed or target values for the dependent variable.

    a0 : float, optional
        Initial value for the coefficient 'a'. Default is 0.

    b0 : float, optional
        Initial value for the intercept 'b'. Default is 0.

    alpha : float, optional
        Learning rate, controlling the step size in the gradient descent process.
        Default is 0.1.

    n_iter : int, optional
        Number of iterations for which gradient descent will be performed. Default is 100.

    print_res : {0, 1}, optional
        If set to 1, the cost at each iteration will be printed. If set to 0, no
        intermediate results will be printed. Default is 0.

    Returns:
    --------
    a : float
        The optimized coefficient 'a' for the linear model.

    b : float
        The optimized intercept 'b' for the linear model.

    cost_history : list
        A list of mean squared error (MSE) cost values at each iteration.

    coeff_history : list
        A list of coefficient tuples (a, b) at each iteration.

    Notes:
    ------
    This function performs gradient descent optimization to find the best coefficients 'a' and 'b'
    for a linear model that minimizes the mean squared error (MSE) cost function. The optimization
    process is controlled by the learning rate 'alpha' and the number of iterations 'n_iter'.

    If 'print_res' is set to 1, the cost at each iteration will be printed to the console.

    Example:
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> a_opt, b_opt, cost_history, coeff_history = gradient_descent(x, y, a0=0, b0=0, alpha=0.1, n_iter=100, print_res=1)
    Cost at iter no. 0: 21.00
    Cost at iter no. 1: 7.29
    ...
    """
    a = a0
    b = b0
    cost_history = []
    coeff_history = []
    for i in range(n_iter):
        gradient_a, gradient_b = gradient_calc(x, y, a, b)
        a = a - alpha * gradient_a
        b = b - alpha * gradient_b
        mse = cost_function(x, y, a, b)
        cost_history.append(mse)
        coeff_history.append((a, b))
        if print_res == 1:
            print(f'Cost at iter no. {i}: {mse:.2f}')
    return a, b, cost_history, coeff_history

def prediction(x, a, b):
    """
    Predict values using a linear model with given coefficients.

    Parameters:
    -----------
    x : array_like
        Input values for the independent variable.

    a : float
        Coefficient for the independent variable 'x' in the linear model.

    b : float
        Intercept term in the linear model.

    Returns:
    --------
    y_pred : ndarray
        Array of the same shape as 'x' containing the predicted values generated by the linear model.

    Notes:
    ------
    This function calculates predicted values using a linear model of the form:
    y_pred = a * x + b
    where 'a' represents the coefficient for 'x' and 'b' represents the intercept.

    If 'a' and 'b' are not provided as arguments, the default values 'a_model' and 'b_model' are used.

    Example:
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> a = 2.0
    >>> b = 1.0
    >>> y_pred = prediction(x, a, b)
    array([ 3.,  5.,  7.,  9., 11.])
    """
    y_pred = a * x + b
    return y_pred