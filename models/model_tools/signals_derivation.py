

def compute_discrete_derivative(signal, order, dt):
    if order == 1:
        return (signal[1:] - signal[:-1]) / dt
    elif order == 2:
        first_deriv = (signal[1:] - signal[:-1]) / dt
        return (first_deriv[1:] - first_deriv[:-1]) / dt
