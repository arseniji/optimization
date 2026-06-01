import numpy as np


def gradient_descent_with_constant_step(
        f, grad,
        x0, y0,
        step0=0.01,
        eps0=1e-4,
        eps1=1e-5,
        eps2=1e-8,
        max_iter=10000,
        min_step=1e-12
):

    x, y = x0, y0
    k = 0
    first_condition = False
    temp = [(k, x, y, f(x, y), np.linalg.norm(grad(x, y)))]

    while k < max_iter:
        g = grad(x, y)
        grad_norm = np.linalg.norm(g)
        f_current = f(x, y)

        # Критерий останова по норме градиента
        if grad_norm < eps1:
            break

        # Дробление шага: СБРАСЫВАЕМ t = step0 на каждой итерации
        t = step0
        x_new, y_new, f_new = x, y, f_current
        step_found = False

        while t >= min_step:
            x_new = x - t * g[0]
            y_new = y - t * g[1]
            f_new = f(x_new, y_new)
            # Условие достаточного убывания Армихо
            if f_new < f_current - eps0 * t * grad_norm ** 2:
                step_found = True
                break
            t /= 2

        if not step_found:
            # Шаг слишком маленький — остановка
            temp.append((k + 1, x, y, f(x, y), grad_norm))
            break

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
        k += 1
        temp.append((k, x, y, f(x, y), np.linalg.norm(grad(x, y))))

    return x, y, k, temp