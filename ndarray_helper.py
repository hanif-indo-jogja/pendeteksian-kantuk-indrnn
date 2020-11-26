import numpy as np

def rotate_left(arr, n_rotate):
    act_n_rotate = n_rotate % 4
    if (act_n_rotate == 0):
        return arr
    arr_length = len(arr)
    
    return np.concatenate((arr[act_n_rotate:arr_length], arr[0:act_n_rotate]))
