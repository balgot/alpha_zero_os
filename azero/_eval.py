# Modified from `open_spiel` repo: https://github.com/deepmind/open_spiel
"""An MCTS Evaluator for an AlphaZero model."""

import numpy as np

from open_spiel.python.algorithms import mcts
import pyspiel
from open_spiel.python.utils import lru_cache


class AlphaZeroEvaluator(mcts.Evaluator):
    """An AlphaZero MCTS Evaluator."""

    def __init__(self, game, model, cache_size=2**16):
        """An AlphaZero MCTS Evaluator."""
        if game.num_players() > 2:
            raise ValueError("Game must be for (at most) two players.")
        game_type = game.get_type()
        if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
            raise ValueError("Game must have terminal rewards.")
        if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError("Game must have sequential turns.")

        self._model = model
        self._cache = lru_cache.LRUCache(cache_size)

    def cache_info(self):
        return self._cache.info()

    def clear_cache(self):
        self._cache.clear()

    def _inference(self, state):
        # Make a singleton batch
        obs = np.expand_dims(state.observation_tensor(), 0)
        mask = np.expand_dims(state.legal_actions_mask(), 0)

        # ndarray isn't hashable
        cache_key = obs.tobytes() + mask.tobytes()

        value, policy = self._cache.make(cache_key, lambda: self._model.inference(obs, mask))
        return value[0, 0], policy[0]  # Unpack batch

    def evaluate(self, state):
        """Returns a value for the given state."""
        value, _ = self._inference(state)
        return np.array([value, -value])

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            # Returns the probabilities for all actions.
            _, policy = self._inference(state)
            return [(action, policy[action]) for action in state.legal_actions()]
