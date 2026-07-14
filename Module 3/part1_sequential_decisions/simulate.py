from agents import Agent
from environments import Environment
from plots import animate_frames


def train_agent(agent: Agent, env: Environment, num_episodes: int = 6_000) -> list:
    """Train a RL agent in an Environment."""
    reward_history = []
    early_stop_threshold = 195
    early_stop_consecutive = 100
    consecutive_solves = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update_q_table(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        agent.update_epsilon()

        reward_history.append(total_reward)
        print(f"Episode {episode} reward: {total_reward}", end="\r")

        if total_reward >= early_stop_threshold:
            consecutive_solves += 1
            if consecutive_solves >= early_stop_consecutive:
                print(f"\nEnvironment solved in {episode + 1} episodes!")
                break
        else:
            consecutive_solves = 0

    print("\nTraining completed")
    return reward_history


def render_agent_performance(
    agent: Agent,
    env: Environment,
    num_episodes: int = 3,
    fps: int = 10,
    max_frames: int = 500,
) -> None:
    """Watch a trained agent act in an Environment.

    If the env was created with render_mode="rgb_array", we collect the frames and
    play them back as an animation inside the notebook, which also works on Colab.
    """
    frames = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # in rgb_array mode this hands us back a frame, in human mode it draws
            # to a window and gives us None
            frame = env.render()
            if frame is not None and len(frames) < max_frames:
                frames.append(frame)

            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        print(f"Episode {episode + 1} completed with reward {reward}")

    env.close()
    animate_frames(frames, fps=fps)
