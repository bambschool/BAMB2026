import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import HTML, display


def animate_frames(frames: list, fps: int = 10) -> None:
    """Play a list of rgb_array frames as an inline animation.

    This works both locally and on Google Colab, unlike render_mode="human",
    which needs a display that Colab does not have.
    """
    if not frames:
        return

    # keep the embedded animation small enough for the notebook to stay happy
    plt.rcParams["animation.embed_limit"] = 100  # MB

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axis("off")
    image = ax.imshow(frames[0])

    def update(frame_index: int):
        image.set_data(frames[frame_index])
        return (image,)

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=True
    )

    # close the figure so the last frame isn't also shown as a static image
    plt.close(fig)
    display(HTML(anim.to_jshtml()))


def plot_frozenlake_environment(env):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    env.reset()
    ax.imshow(env.render())
    ax.axis("off")
    ax.set_title("FrozenLake environment")
    plt.show()


def calculate_moving_average(reward_history, window_size=100):
    cumsum = np.cumsum(np.insert(reward_history, 0, 0))
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    # Use the first calculated average for all previous elements
    first_avg = moving_avg[0]
    padded_avg = np.full(window_size - 1, first_avg)

    return np.concatenate([padded_avg, moving_avg])


def plot_performance(reward_history):
    plt.figure(figsize=(6, 4))
    plt.plot(reward_history)
    plt.plot(calculate_moving_average(reward_history))
    plt.title("Performance over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()


def plot_performance_comparison(rewards_1, label_1, rewards_2, label_2):
    # Plot results
    plt.figure(figsize=(6, 4))
    plt.plot(calculate_moving_average(rewards_1), label=label_1)
    plt.plot(calculate_moving_average(rewards_2), label=label_2)
    plt.title(f"{label_1} vs {label_2} Performance")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.show()


def qtable_directions_map(qtable, map_size) -> tuple:
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, env, map_size) -> None:
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    plt.show()
