import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any

from jaxued.environments import UnderspecifiedEnv

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info
        
class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree_map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int

class LogWrapper(UnderspecifiedEnv):
    """Log the episode returns, lengths and achievements."""

    def __init__(self, env):
        self._env = env
    
    @property
    def default_params(self):
        return self._env.default_params

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state
    

    @partial(jax.jit, static_argnums=(0, 3))
    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level,
        params
    ):
        state = LogEnvState(level, 0.0, 0, 0.0, 0, 0)
        return self.get_obs(state), state
    
    @partial(jax.jit, static_argnums=(0, 4))
    def step_env(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params= None,
    ):
        obs, env_state, reward, done, info = self._env.step_env(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"]    = state.returned_episode_returns
        info["returned_episode_lengths"]    = state.returned_episode_lengths
        info["timestep"]                    = state.timestep
        info["returned_episode"]            = done
        # info['achievements']                = env_state.achievements
        # info['achievement_count']           = env_state.achievements.sum()
        
        if hasattr(env_state, 'player_level'):
            info['floor']                       = env_state.player_level
        return obs, state, reward, done, info

    def get_obs(self, state: LogEnvState) -> chex.Array:
        return self._env.get_obs(state.env_state)

    def action_space(self, params) -> Any:
        return self._env.action_space(params)

# below is adapted from jaxued 
# https://github.com/DramaCow/jaxued/

@struct.dataclass
class EnvState:
    pass

@struct.dataclass
class Level:
    pass

@struct.dataclass
class AutoReplayState:
    env_state: EnvState
    level: Level

    level_idx: int = -1


class DistResetEnvWrapper(AutoResetEnvWrapper):

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        state = AutoReplayState(env_state=env_state, level=env_state.env_state)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params = None,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        """
        Step with "auto replay"
        """
        rng_reset, rng_step = jax.random.split(rng)
        obs_re, env_state_re = self._env.reset_env_to_level(rng_reset, state.env_state.env_state, params)
        obs_st, env_state_st, reward, done, info = self._env.step(
            rng_step, state.env_state, action, params
        )
        env_state = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), env_state_re, env_state_st)
        obs = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)
        return obs, state.replace(env_state=env_state), reward, done, info

    @partial(jax.jit, static_argnums=(0, 6))
    def step_with_dist_reset(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        y: chex.Array, 
        levels,
        params = None,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        rng_reset, rng_step = jax.random.split(rng)

        rng_reset, _rng = jax.random.split(rng_reset)
        level_idx = jax.random.choice(_rng, len(y), p=y)
        level = jax.tree_util.tree_map(lambda x: x[level_idx], levels)

        obs_re, env_state_re = self._env.reset_env_to_level(rng_reset, level, params)
        obs_st, env_state_st, reward, done, info = self._env.step(
            rng_step, state.env_state, action, params
        )
        env_state = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), env_state_re, env_state_st)
        obs = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)
        return obs, state.replace(env_state=env_state), reward, done, info

    @partial(jax.jit, static_argnums=(0, 3))
    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
        params = None
    ):
        obs, env_state = self._env.reset_env_to_level(rng, level, params)
        return obs, AutoReplayState(env_state=env_state, level=level)


class LearnabilityGradWrapper(DistResetEnvWrapper):
    @partial(jax.jit, static_argnums=(0, 6))
    def step_with_dist_reset(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        y: chex.Array, 
        levels,
        ret_table,
        dones_table,
        params = None,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
    
        rng_reset, rng_step = jax.random.split(rng)

        rng_reset, _rng = jax.random.split(rng_reset)
        level_idx = jax.random.choice(_rng, len(y), p=y)
        level = jax.tree_util.tree_map(lambda x: x[level_idx], levels)

        obs_re, env_state_re = self._env.reset_env_to_level(rng_reset, level, params)
        obs_st, env_state_st, reward, done, info = self._env.step(
            rng_step, state.env_state, action, params
        )
        env_state = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), env_state_re, env_state_st)
        obs = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)

        # remember to update wrapper
        ret = info["episode_return"]

        state = state.replace(
            env_state=env_state, 
            grad = grad, 
            level_idx = jax.lax.select(done, level_idx, state.level_idx)
        )

        return obs, state, reward, done, info, ret