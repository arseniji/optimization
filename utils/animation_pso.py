import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def create_swarm_animation(f, bounds, snapshots):
    res = 80
    xs = np.linspace(bounds[0], bounds[1], res)
    ys = np.linspace(bounds[0], bounds[1], res)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)

    # Рендерим фон один раз через matplotlib → base64 PNG
    fig_bg, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(X, Y, Z, levels=40, cmap="viridis")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.axis("off")
    fig_bg.tight_layout(pad=0)

    buf = io.BytesIO()
    fig_bg.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                   facecolor="black", dpi=120)
    plt.close(fig_bg)
    buf.seek(0)
    bg_b64 = base64.b64encode(buf.read()).decode()
    bg_uri = f"data:image/png;base64,{bg_b64}"

    first = snapshots[0]

    frames = [
        go.Frame(
            data=[
                go.Scatter(
                    x=snap["xs"], y=snap["ys"],
                    mode="markers",
                    marker=dict(size=6, color="#38bdf8", opacity=0.75),
                ),
                go.Scatter(
                    x=[snap["gbest_x"]], y=[snap["gbest_y"]],
                    mode="markers",
                    marker=dict(size=14, color="red", symbol="star"),
                ),
            ],
            name=str(snap["iter"]),
            layout=go.Layout(
                title=dict(
                    text=f"Итерация {snap['iter']} | gbest = {snap['gbest_val']:.6f}"
                )
            ),
        )
        for snap in snapshots
    ]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=first["xs"], y=first["ys"],
                mode="markers",
                marker=dict(size=6, color="#38bdf8", opacity=0.75),
                name="Частицы",
            ),
            go.Scatter(
                x=[first["gbest_x"]], y=[first["gbest_y"]],
                mode="markers",
                marker=dict(size=14, color="red", symbol="star"),
                name="gbest",
            ),
        ],
        frames=frames,
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=550,
            margin=dict(l=10, r=10, t=50, b=10),
            title=dict(
                text=f"Итерация 0 | gbest = {first['gbest_val']:.6f}"
            ),
            xaxis=dict(
                range=[bounds[0], bounds[1]],
                showgrid=False, zeroline=False, visible=False,
            ),
            yaxis=dict(
                range=[bounds[0], bounds[1]],
                scaleanchor="x", showgrid=False, zeroline=False, visible=False,
            ),
            # Фон — статичная картинка, не участвует в анимации
            images=[dict(
                source=bg_uri,
                xref="x", yref="y",
                x=bounds[0], y=bounds[1],
                sizex=bounds[1] - bounds[0],
                sizey=bounds[1] - bounds[0],
                sizing="stretch",
                layer="below",
                opacity=1.0,
            )],
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.12, x=0, xanchor="left",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 80, "redraw": False},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        }],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        }],
                    ),
                ],
            )],
            sliders=[dict(
                active=0,
                currentvalue=dict(
                    prefix="Итерация: ", visible=True, xanchor="right"
                ),
                pad=dict(t=10),
                steps=[
                    dict(
                        method="animate",
                        label=str(snap["iter"]),
                        args=[[str(snap["iter"])], {
                            "frame": {"duration": 80, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        }],
                    )
                    for snap in snapshots
                ],
            )],
        ),
    )

    return fig.to_html(full_html=False)