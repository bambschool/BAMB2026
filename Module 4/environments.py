"""
environments.py: GridWorld environments 
Prepared by Charley M. Wu (TU Darmstadt; hmc-lab.com) 
for the 2026 Barcelona Summer School for Advanced Modeling of Behavior (BAMB; https://www.bambschool.org/)
"""

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Union
import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """Minimal deterministic gridworld environment for DYNA-Q tutorials."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    ACTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }

    def __init__(
        self,
        size: int = 4,
        max_steps: int = 50,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Optional[Tuple[int, int]] = None,
        obstacles: Optional[Iterable[Tuple[int, int]]] = None,
        render_mode: Optional[str] = "human",
    ):
        self.size = size
        self.max_steps = max_steps
        self.start_pos = start_pos
        self.goal_pos = goal_pos if goal_pos is not None else (size - 1, size - 1)
        self.obstacles = set(obstacles or [])
        self.render_mode = render_mode

        if self.start_pos == self.goal_pos:
            raise ValueError("start_pos and goal_pos must be different")

        if self.start_pos in self.obstacles or self.goal_pos in self.obstacles:
            raise ValueError("start_pos and goal_pos cannot be on an obstacle")

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Discrete(size * size)

        self.state = self._pos_to_state(self.start_pos)
        self.step_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = self._pos_to_state(self.start_pos)
        self.step_count = 0
        return self.state, {}

    def step(self, action: int):
        self.step_count += 1
        next_state, reward, terminated = self.transition(self.state, action)
        truncated = self.step_count >= self.max_steps and not terminated
        self.state = next_state
        return next_state, reward, terminated, truncated, {}

    def transition(self, state: int, action: int):
        """Returns the next state, reward, and terminal flag for a state/action."""
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        position = self._state_to_pos(state)
        delta = self.ACTIONS[action]
        candidate = (position[0] + delta[0], position[1] + delta[1])

        if self._is_valid_position(candidate):
            position = candidate

        next_state = self._pos_to_state(position)
        terminated = position == self.goal_pos
        reward = 1.0 if terminated else -0.01
        return next_state, reward, terminated

    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        for ox, oy in self.obstacles:
            grid[ox][oy] = "#"

        gx, gy = self.goal_pos
        grid[gx][gy] = "G"

        ax, ay = self._state_to_pos(self.state)
        grid[ax][ay] = "A"

        output = "\n" + "\n".join(" ".join(row) for row in grid)
        if self.render_mode == "human":
            sys.stdout.write(output + "\n")
            sys.stdout.flush()
            return None
        return output

    def render_policy(self, q_values: np.ndarray):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        for ox, oy in self.obstacles:
            grid[ox][oy] = "#"

        gx, gy = self.goal_pos
        grid[gx][gy] = "G"

        for state in range(self.size * self.size):
            pos = self._state_to_pos(state)
            if pos == self.goal_pos or pos in self.obstacles:
                continue
            best_action = int(np.argmax(q_values[state]))
            symbol = {0: "↑", 1: "→", 2: "↓", 3: "←"}[best_action]
            grid[pos[0]][pos[1]] = symbol

        output = "\n" + "\n".join(" ".join(row) for row in grid)
        if self.render_mode == "human":
            sys.stdout.write(output + "\n")
            sys.stdout.flush()
            return None
        return output

    def describe(self):
        description = {
            "size": self.size,
            "start_pos": self.start_pos,
            "goal_pos": self.goal_pos,
            "obstacles": sorted(self.obstacles),
            "max_steps": self.max_steps,
        }
        print(description)

    def set_start_pos(self, start_pos: Tuple[int, int]):
        self.start_pos = start_pos
        self.state = self._pos_to_state(start_pos)

    def set_goal_pos(self, goal_pos: Tuple[int, int]):
        if goal_pos in self.obstacles:
            raise ValueError("Goal position cannot overlap an obstacle")
        self.goal_pos = goal_pos

    def set_obstacles(self, obstacles: Iterable[Tuple[int, int]]):
        obstacles = set(obstacles or [])
        if self.start_pos in obstacles or self.goal_pos in obstacles:
            raise ValueError("Obstacles cannot overlap start or goal positions")
        self.obstacles = obstacles

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.size + pos[1]

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        return (
            0 <= pos[0] < self.size
            and 0 <= pos[1] < self.size
            and pos not in self.obstacles
        )

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return self.action_space.n



gym.envs.registration.register(
    id="SimpleGridWorld-v0",
    entry_point="environments:GridWorldEnv",
)


@dataclass
class EnvironmentConfig:
    name: str = "default"
    size: int = 4
    start_pos: Tuple[int, int] = (0, 0)
    goal_pos: Optional[Tuple[int, int]] = None
    obstacles: List[Tuple[int, int]] = field(default_factory=list)
    max_steps: int = 50

    @classmethod
    def from_ascii_map(
        cls,
        ascii_map: Union[str, Iterable[str]],
        name: str = "ascii_grid",
        max_steps: int = 50,
    ) -> "EnvironmentConfig":
        size, start_pos, goal_pos, obstacles = _parse_ascii_map(ascii_map)
        return cls(
            name=name,
            size=size,
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacles=obstacles,
            max_steps=max_steps,
        )

    def build(self) -> GridWorldEnv:
        return GridWorldEnv(
            size=self.size,
            start_pos=self.start_pos,
            goal_pos=self.goal_pos,
            obstacles=self.obstacles,
            max_steps=self.max_steps,
        )


def _parse_ascii_map(ascii_map: Union[str, Iterable[str]]) -> Tuple[int, Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
    if isinstance(ascii_map, str):
        lines = [line for line in ascii_map.splitlines() if line.strip()]
    else:
        lines = [str(line) for line in ascii_map if str(line).strip()]

    if not lines:
        raise ValueError("ASCII map cannot be empty")

    rows: List[List[str]] = []
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) == 1 and len(line.replace(" ", "")) > 1:
            tokens = list(line.strip())
        rows.append(tokens)

    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError("ASCII map rows must all have the same width")

    height = len(rows)
    width = widths.pop()
    if height != width:
        raise ValueError(f"ASCII map must be square, got {height} x {width}")

    start_pos: Optional[Tuple[int, int]] = None
    goal_pos: Optional[Tuple[int, int]] = None
    obstacles: List[Tuple[int, int]] = []

    for i, row in enumerate(rows):
        for j, symbol in enumerate(row):
            symbol = symbol.strip()
            if not symbol or symbol == ".":
                continue
            if symbol == "#":
                obstacles.append((i, j))
            elif symbol in {"A", "S"}:
                if start_pos is not None:
                    raise ValueError("ASCII map must contain exactly one start position")
                start_pos = (i, j)
            elif symbol == "G":
                if goal_pos is not None:
                    raise ValueError("ASCII map must contain exactly one goal position")
                goal_pos = (i, j)
            else:
                raise ValueError(f"Unrecognized ASCII map symbol '{symbol}' at row {i}, col {j}")

    if start_pos is None:
        raise ValueError("ASCII map must contain exactly one start position ('A' or 'S')")
    if goal_pos is None:
        raise ValueError("ASCII map must contain exactly one goal position ('G')")

    return height, start_pos, goal_pos, obstacles


def describe_config(config: EnvironmentConfig) -> None:
    print("Environment configuration:")
    print(f"  name: {config.name}")
    print(f"  size: {config.size}")
    print(f"  start_pos: {config.start_pos}")
    print(f"  goal_pos: {config.goal_pos or (config.size - 1, config.size - 1)}")
    print(f"  obstacles: {config.obstacles}")
    print(f"  max_steps: {config.max_steps}")


def visualize_config(config: EnvironmentConfig) -> GridWorldEnv:
    env = config.build()
    env.describe()
    env.render()
    return env


def generate_environment_variants() -> List[EnvironmentConfig]:
    return [
        EnvironmentConfig(
            name="open_grid",
            size=4,
            start_pos=(0, 0),
            goal_pos=(3, 3),
            obstacles=[],
            max_steps=50,
        ),
        EnvironmentConfig(
            name="single_wall",
            size=4,
            start_pos=(0, 0),
            goal_pos=(3, 3),
            obstacles=[(1, 0), (1, 1), (1, 3)],
            max_steps=60,
        ),
        EnvironmentConfig(
            name="dead_end_maze",
            size=5,
            start_pos=(0, 0),
            goal_pos=(4, 4),
            obstacles=[(1, 0), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3)],
            max_steps=80,
        ),
        EnvironmentConfig(
            name="narrow_corridor",
            size=5,
            start_pos=(0, 0),
            goal_pos=(4, 4),
            obstacles=[(0, 2), (1, 2), (3, 2), (4, 2)],
            max_steps=80,
        ),
        EnvironmentConfig(
            name="branching_maze",
            size=5,
            start_pos=(0, 0),
            goal_pos=(4, 4),
            obstacles=[(1, 0), (1, 1), (1, 3), (2, 3), (3, 3)],
            max_steps=80,
        ),
    ]

def list_environment_variants():
    """List all prebuilt environment variants."""
    variants = generate_environment_variants()
    for i, variant in enumerate(variants):
        print(f"{i}: {variant.name} | size={variant.size}, start={variant.start_pos}, goal={variant.goal_pos}")
    return variants

def choose_environment(variants, index):
    """Choose a prebuilt variant by index."""
    if index < 0 or index >= len(variants):
        raise IndexError("Invalid environment variant index")
    return variants[index]

def create_custom_environment(
    size: int = 4,
    start_pos: Tuple[int, int] = (0, 0),
    goal_pos=None,
    obstacles=None,
    max_steps: int = 50,
    ascii_map: Optional[Union[str, Iterable[str]]] = None,
):
    """Create a custom EnvironmentConfig with specified parameters or from an ASCII map."""
    if ascii_map is not None:
        config = EnvironmentConfig.from_ascii_map(ascii_map, name="custom-ascii", max_steps=max_steps)
        return config

    if goal_pos is None:
        goal_pos = (size - 1, size - 1)
    return EnvironmentConfig(
        name=f"custom-{size}x{size}",
        size=size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacles=obstacles or [],
        max_steps=max_steps,
    )


def build_and_show_environment(variant):
    """Build an EnvironmentConfig and visualize it."""
    env = variant.build()
    env.describe()
    env.render()
    return env

def evaluate_config_heuristic(config: EnvironmentConfig) -> float:
    cell_count = config.size * config.size
    obstacle_density = len(config.obstacles) / float(cell_count)
    corridor_bonus = 0.0
    if config.size >= 4 and len(config.obstacles) > 0:
        corridor_bonus = 0.2
    return obstacle_density + corridor_bonus

def generate_sticky_env_sequence(env_names: List[str], episodes: int, stay_prob: float = 0.9, seed: int = 0) -> List[str]:
    np.random.seed(seed)
    env_sequence: List[str] = []
    current = random.choice(env_names)
    for _ in range(episodes):
        if np.random.rand() > stay_prob:
            choices = [name for name in env_names if name != current]
            current = random.choice(choices)
        env_sequence.append(current)
    return env_sequence

