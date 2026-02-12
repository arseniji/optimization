import numpy as np

def center_grad(f,x,y,h = 1e-5):
    df_dx = (f(x + h,y) - f(x - h,y)) / (2*h)
    df_dy = (f(x,y + h) - f(x,y-h)) / (2*h)
    return np.array([df_dx, df_dy])