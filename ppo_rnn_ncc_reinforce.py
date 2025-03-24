import argparse
import os
import sys

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
from typing import Any

from flax.training import orbax_utils
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

import wandb
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import functools

from ued_wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
    DistResetEnvWrapper,
    LearnabilityGradWrapper
)
from logz.batch_logging import create_log_dict, batch_log

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.world_gen.world_gen import generate_world

from ncc_utils import scale_y_by_ti_ada, ScaleByTiAdaState, ti_ada, projection_simplex_truncated

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl

class CustomTrainState(TrainState):
    levels: Any
    y: jnp.ndarray
    y_opt_state: ScaleByTiAdaState

    ret_table: jnp.ndarray
    dones_table: jnp.ndarray

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Create environment
    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    DEFAULT_STATICS = env.default_static_params()
    env_params = env.default_params
    sample_random_level = lambda rng: generate_world(rng, env.default_params, DEFAULT_STATICS)

    # Wrap with some extra logging
    log_env = LogWrapper(env)

    # Wrap with a batcher, maybe using optimistic resets
    reset_env = AutoResetEnvWrapper(log_env)
    # dist_env = DistResetEnvWrapper(log_env)
    dist_env = LearnabilityGradWrapper(log_env)
    score_env = BatchEnvWrapper(log_env, num_envs=config["BUFFER_SIZE"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env_params).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )
        network_params = network.init(_rng, init_hstate, init_x)
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            ti_ada(vy0 = jnp.zeros(config["BUFFER_SIZE"]), eta=linear_schedule)
        )

        # tx = optax.chain(
        #     optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        #     optax.adam(learning_rate=linear_schedule, eps=1e-5),
        # )
        y_ti_ada = scale_y_by_ti_ada(eta=config["META_LR"])

        # init tables
        dones_table = jnp.zeros(config["BUFFER_SIZE"], dtype=int)
        ret_table = jnp.zeros((config["BUFFER_SIZE"], config["WINDOW_SIZE"]))

        rng, _rng = jax.random.split(rng)
        levels = jax.vmap(sample_random_level)(jax.random.split(_rng, config["BUFFER_SIZE"]))
        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            y = jnp.full(config["BUFFER_SIZE"], 1 / config["BUFFER_SIZE"]),
            y_opt_state = y_ti_ada.init(jnp.zeros(config["BUFFER_SIZE"])),
            levels=levels,
            ret_table = ret_table,
            dones_table = dones_table
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        level_idxs = jax.random.choice(_rng, config["BUFFER_SIZE"], shape=(config["NUM_ENVS"], ))
        levels = jax.tree_util.tree_map(lambda l: l[level_idxs], train_state.levels)

        rng, _rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(dist_env.reset_env_to_level, in_axes=(0, 0, None))(jax.random.split(_rng, config["NUM_ENVS"]), levels, env_params)
        env_state = env_state.replace(level_idx=level_idxs)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    rng,
                    update_step,
                ) = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])

                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info, ret =  jax.vmap(dist_env.step_with_dist_reset, in_axes=(0, 0, 0, None, None, None))(
                    jax.random.split(_rng, config["NUM_ENVS"]), env_state, action, train_state.y, train_state.levels, env_params
                )
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )

                # update table
                def _update_loop(tables, data):
                    ret_table, dones_table = tables
                    done, ret, level_idx = data

                    new_history = jax.lax.select(dones_table[level_idx] != config["WINDOW_SIZE"], ret_table[level_idx].at[dones_table[level_idx]].set(ret), jnp.roll(ret_table[level_idx], -1).at[-1].set(ret))

                    history = jax.lax.select(done, new_history, ret_table[level_idx])
                    ret_table = ret_table.at[level_idx].set(history)
                    dones_table = dones_table.at[level_idx].add(done)

                    return (ret_table, dones_table), None

                (ret_table, dones_table), _ = jax.lax.scan(_update_loop, (train_state.ret_table, train_state.dones_table), xs = (done, ret, env_state.level_idx))
                train_state = train_state.replace(
                    ret_table = ret_table,
                    dones_table = dones_table,
                )

                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done,
                    hstate,
                    rng,
                    update_step,
                )

                return runner_state, transition


            def score_fn(train_state, rng):

                def score_env_step(runner_state, unused):
                    (
                        env_state,
                        last_obs,
                        last_done,
                        hstate,
                        rng,

                    ) = runner_state
                    rng, _rng = jax.random.split(rng)

                    # SELECT ACTION
                    ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                    hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                    value, action, log_prob = (
                        value.squeeze(0),
                        action.squeeze(0),
                        log_prob.squeeze(0),
                    )

                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    obsv, env_state, reward, done, info = score_env.step(
                        _rng, env_state, action, env_params
                    )
                    transition = Transition(
                        last_done, action, value, reward, log_prob, last_obs, info
                    )
                    runner_state = (
                        env_state,
                        obsv,
                        done,
                        hstate,
                        rng,
                    )
                    return runner_state, transition

                rng, _rng = jax.random.split(rng)
                init_obs, env_state = jax.vmap(score_env.reset_env_to_level, in_axes=(0, 0, None))(jax.random.split(_rng, config["BUFFER_SIZE"]), train_state.levels, env_params)
                init_done = jnp.zeros((config["BUFFER_SIZE"]), dtype=bool)
                init_hs = ScannedRNN.initialize_carry(
                    config["BUFFER_SIZE"], config["LAYER_SIZE"]
                )  # (n_envs, hs_size)
                rng, _rng = jax.random.split(rng)
                step_state = (
                    env_state,
                    init_obs,
                    init_done,
                    init_hs,
                    _rng,
                )
                
                _, transition = jax.lax.scan(
                    score_env_step, step_state, None, 1000
                )

                mask = transition.done.cumsum(axis=0) < 1
                level_returns = (transition.reward * mask).sum(axis=0)
                
                return level_returns

             ### NCC UPDATES ###
            def update_y(rng):
                
                returns = train_state.ret_table.T
                ret_mask = jax.vmap(lambda num_dones: jnp.arange(config["WINDOW_SIZE"]) < num_dones)(train_state.dones_table).T
                level_mask = (train_state.dones_table > 2)

                def gaussian_pdf(x, mean, var):
                    return (1 / (jnp.sqrt(2 * jnp.pi) * var)) * jnp.exp(-0.5 * ((x - mean) / var) ** 2)
                
                mean_returns = jnp.mean(returns, axis=0, where=ret_mask)
                mu = mean_returns.mean(where=level_mask)
                sigma_2 = mean_returns.var(where=level_mask)
                
                pdf_values = jnp.where(level_mask, gaussian_pdf(mean_returns, mu, sigma_2), 0.0)
        
                scores = jnp.where(level_mask, jnp.sqrt(returns.var(axis=0, where=level_mask)) / jnp.sqrt(train_state.dones_table) * pdf_values, 0.0)

                y_fn = lambda y: y.T @ scores - config["META_REG"] * jnp.log(y + 1e-6).T @ y
                grad, y_opt_state = y_ti_ada.update(jax.grad(y_fn)(train_state.y), train_state.y_opt_state)
                y = projection_simplex_truncated(train_state.y + grad, config["META_TRUNC"])

                return train_state.replace(
                    y = y,
                    y_opt_state = y_opt_state
                )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
            ) = runner_state

            rng, _rng = jax.random.split(rng)
            # train_state = jax.lax.cond(
            #     unused % 16 == 0, update_y, lambda r: train_state, _rng
            # )
            train_state = update_y(_rng)

            initial_hstate = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, (train_state, *runner_state[1:]), None, config["NUM_STEPS"]
            )

            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
            ) = runner_state

            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    )
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            metric["adv_entropy"] = train_state.y.T @ jnp.log(train_state.y + 1e-6)

            rng = update_state[-1]
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config, adv_entropy=True)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(callback, metric, update_step)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"]), config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-PPO_RNN-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                config["TOTAL_TIMESTEPS"],
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=np.random.randint(2**31))
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_policy", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str, default="nate_jaxued")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # NCC Stuff
    parser.add_argument("--buffer_size", type=int, default=4000)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--meta_trunc", type=float, default=1e-5)
    parser.add_argument("--meta_reg", type=float, default=0.1)
    parser.add_argument("--meta_lr", type=float, default=1e-2)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)
