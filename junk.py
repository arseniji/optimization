import numpy as np
from fontTools.misc.bezierTools import epsilon
import matplotlib.pyplot as plt
from matplotlib import cm


def f(x,y):
    return - np.cos(x)* np.cos(y) * np.exp(-((x - np.pi)**2 + (y - np.pi)**2))
def grad (x,y,h =0.0000001):
    df_dx = (f(x + h,y) - f(x - h,y)) / (2*h)
    df_dy = (f(x,y + h) - f(x,y-h)) / (2*h)
    return np.array([df_dx, df_dy])

def gradient_descent_with_constant_step(
        x0, y0, step0 = 0.2,
        eps0 = 0.1,
        eps1 = 1e-5,
        eps2 = 1e-6,
        max_iter = 10000,
        min_step = 1e-10):
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
            if (delta_f < 0) or (abs(delta_f) < epsilon * grad_norm**2): break
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

def plot_graph(temp,x_range=(0, 6), y_range=(0, 6), title = "ГРадиент"):

    xs = [h[1] for h in temp]
    ys = [h[2] for h in temp]
    zs = [h[3] for h in temp]

    x = np.linspace(x_range[0], x_range[1], 30)
    y = np.linspace(y_range[0], y_range[1], 30)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Строим поверхность функции
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.plasma,  # Красивая цветовая схема
        alpha=0.85,  # Прозрачность для видимости траектории
        linewidth=0,
        antialiased=False,
        zorder=1
    )

    # Добавляем траекторию спуска
    ax.plot(
        xs, ys, zs,
        color='white',  # Белая линия для контраста
        linewidth=3.0,
        marker='o',
        markersize=6,
        markevery=1,  # Маркеры на каждой точке
        markerfacecolor='red',
        markeredgecolor='darkred',
        markeredgewidth=1.5,
        label='Траектория спуска',
        zorder=10
    )

    # Начальная точка (зелёная)
    ax.scatter(
        [xs[0]], [ys[0]], [zs[0]],
        color='limegreen',
        s=200,
        edgecolors='darkgreen',
        linewidths=2,
        label=f'Старт: ({xs[0]:.2f}, {ys[0]:.2f})',
        zorder=20
    )

    # Конечная точка (красная)
    ax.scatter(
        [xs[-1]], [ys[-1]], [zs[-1]],
        color='red',
        s=200,
        edgecolors='darkred',
        linewidths=2,
        label=f'Финиш: ({xs[-1]:.2f}, {ys[-1]:.2f})',
        zorder=20
    )

    # Глобальный минимум (золотая звезда)
    ax.scatter(
        [np.pi], [np.pi], [f(np.pi, np.pi)],
        color='gold',
        s=300,
        marker='*',
        edgecolors='orange',
        linewidths=2,
        label='Глобальный минимум (π, π)',
        zorder=30
    )

    # Настройка осей
    ax.set_xlabel('X', fontsize=14, labelpad=12)
    ax.set_ylabel('Y', fontsize=14, labelpad=12)
    ax.set_zlabel('f(X, Y)', fontsize=14, labelpad=12)
    ax.set_title(title, fontsize=16, pad=20, fontweight='bold')

    # Добавляем легенду
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

    # Цветовая шкала
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.05, aspect=15)
    cbar.set_label('Значение функции', rotation=270, labelpad=25, fontsize=12)

    # Улучшаем внешний вид
    ax.grid(True, alpha=0.3)
    ax.set_box_aspect([1, 1, 0.6])  # Пропорции осей для лучшего восприятия

    plt.tight_layout()

    # Сохраняем график
    safe_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    plt.savefig(f'gradient_descent_3d_{safe_title}.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ 3D-график сохранён: gradient_descent_3d_{safe_title}.png")

    # Показываем интерактивное окно
    plt.show()

    return fig, ax

x,y,k,temp = gradient_descent_with_constant_step(2.5,4.5)

plot_graph(
    temp,
    x_range=(2, 4.5),
    y_range=(2, 4.5),
    title="Градиентный спуск из точки (3.0, 3.0)"
)