import numpy as np

def gradient_descent_with_constant_step(
        f, grad,
        x0, y0, step0 = 0.2,
        eps0 = 0.1,
        eps1 = 1e-5,
        eps2 = 1e-6,
        max_iter = 10000,
        min_step = 1e-10
):
    x ,y = x0, y0
    k = 0
    first_condition = False
    temp = [(k, x, y, f(x,y), np.linalg.norm(grad(x,y)))]
    while k < max_iter:
        g = grad(x,y)
        grad_norm = np.linalg.norm(g)
        f_current = f(x,y)

        if grad_norm < eps1: break

        t = step0
        while True:
            x_new = x - t * g[0]
            y_new = y - t * g[1]
            f_new = f(x_new,y_new)
            delta_f = f_new - f_current
            if (delta_f < 0) or (abs(delta_f) < eps0 * grad_norm**2): break
            t/=2

            if t<min_step:
                print("Шаг очень маленький, остановка")
                temp.append((k+1,x,y,f(x,y),grad_norm))
                return x,y,k+1,temp
        dx = np.linalg.norm([x_new - x, y_new - y])
        df = abs(f_new - f_current)
        second_condition = (dx < eps2) and (df < eps2)

        if first_condition and second_condition:
            x, y = x_new, y_new
            k += 1
            temp.append((k, x, y, f(x, y), np.linalg.norm(grad(x, y))))
            break
        first_condition = second_condition
        x, y = x_new, y_new
        k +=1
        temp.append((k, x, y, f(x, y), np.linalg.norm(grad(x, y))))
    print(f"Начальная точка: ({x0},{y0}), f = {f(x0,y0):.6f}")
    print(f"Найденная точка: ({x},{y}), f = {f(x,y):.6f}")
    print(f"Итерации {k}")
    return x,y,k, temp