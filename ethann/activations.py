import numpy as np

def activation_from_name_or_function(user_input):
    if isinstance(user_input, str):
        if user_input in activation_lookup:
            return activation_lookup[user_input]
        else:
            msg = f'{user_input} is not a valid activation name'
            raise ValueError(msg)
    elif callable(user_input):
        return user_input
    else:
        msg = 'specified activation must be a function or a string'
        raise TypeError(msg)

def relu(vector):
    return np.maximum(vector, 0)

activation_lookup = {'relu': relu}
