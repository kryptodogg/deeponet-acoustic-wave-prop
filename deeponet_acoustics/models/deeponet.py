# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import collections
import json
import os
from pathlib import Path
from functools import partial
from typing import Any, Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from flax import linen as nn
from flax.training import checkpoints
from jax import jit, random, vmap
from jax.typing import ArrayLike
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import soundfile as sf
import time

from deeponet_acoustics.datahandlers.datagenerators import DataInterface
from deeponet_acoustics.models import loss_functions
from deeponet_acoustics.models.datastructures import (
    NetworkArchitectureType,
    TrainingSettings,
    TransferLearning,
)
from deeponet_acoustics.models.networks_flax import (
    flattened_traversal,
    freezeCnnLayersToKeys,
    freezeLayersToKeys,
)
from deeponet_acoustics.utils.timings import TimingsWriter
from deeponet_acoustics.utils.utils import expandCnnData

LossLogger = collections.namedtuple("LossLogger", ["loss_train", "loss_val", "nIter"])


def exponential_decay(step_size, decay_steps, decay_rate, step_offset=0):
    def schedule(i):
        return step_size * decay_rate ** ((i + step_offset) / decay_steps)

    return schedule


TAG_BN = "bn"
TAG_TN = "tn"
TAG_B0 = "b0"
TAG_ADAPTIVE = "adaptive_weights"


# Define the model
class DeepONet:
    is_bn_fnn: bool
    params: flax.core.FrozenDict
    branch_apply: Any
    trunk_apply: Any

    def __init__(
        self,
        settings: TrainingSettings,
        dataset: DataInterface,
        module_bn: tuple[nn.Module, ArrayLike],
        module_tn: tuple[nn.Module, ArrayLike],
        log_dir,
        transfer_learning: TransferLearning | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        lr = settings.learning_rate
        if settings.use_adaptive_weights:
            # adaptive_weights_shape must cover ALL coordinate indices (not just
            # one batch).  coordinate indices from __getitem__ are in [0, dataset.P)
            # per sample.  The original min(batch, N) truncation caused out-of-bounds
            # scatter writes on ROCm → silent exit 139.
            # FIX (2026-06-16): use dataset.P, not min(batch_size, N) * min(batch_size, P).
            adaptive_weights_shape = (dataset.P,)
        else:
            adaptive_weights_shape = []

        decay_steps = settings.decay_steps
        decay_rate = settings.decay_rate

        self.loss_logger = LossLogger([], [], [])
        self.step_offset = 0

        self.log_dir = log_dir
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.is_bn_fnn = module_bn[0].network_type != NetworkArchitectureType.RESNET
        dim_bn = module_bn[1]
        dim_tn = module_tn[1]

        if transfer_learning is None:
            branch_params = module_bn[0].init(
                random.PRNGKey(1234),
                jnp.expand_dims(jnp.ones(dim_bn), axis=0)
                if self.is_bn_fnn
                else expandCnnData(np.ones(dim_bn)),
            )

            trunk_params = module_tn[0].init(random.PRNGKey(4321), jnp.ones(dim_tn))
            if len(adaptive_weights_shape) > 0:
                self.params = flax.core.frozen_dict.freeze(
                    {
                        TAG_BN: branch_params,
                        TAG_TN: trunk_params,
                        TAG_B0: 0.0,
                        TAG_ADAPTIVE: jnp.ones(adaptive_weights_shape),
                    }
                )
            else:
                self.params = flax.core.frozen_dict.freeze(
                    {TAG_BN: branch_params, TAG_TN: trunk_params, TAG_B0: 0.0}
                )

            freeze_layers = set()
        else:
            ckpt_dir = transfer_learning.transfer_model_path
            resume = transfer_learning.resume_learning
            freeze_layers = freezeLayersToKeys(transfer_learning.freeze_layers)

            if resume:
                self.step_offset, self.loss_logger = loadLosses(ckpt_dir)

            # https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html
            self.params = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
            if self.params is None:
                raise Exception(
                    f"Could not load model parameter checkpoint at {ckpt_dir}"
                )

            if len(adaptive_weights_shape) == 0:
                self.params.pop(
                    TAG_ADAPTIVE, None
                )  # remove adaptive weights if existing
            elif TAG_ADAPTIVE in self.params:
                if adaptive_weights_shape != self.params[TAG_ADAPTIVE].shape:
                    raise Exception(
                        "Mismatch between the loaded adaptive weights and the batch dimensions. Reset the batch dimension to equal values as for the transfered model or disable/modify the self-adaptive weights"
                    )
            else:
                self.params[TAG_ADAPTIVE] = jnp.ones(adaptive_weights_shape)

            if not self.is_bn_fnn:
                freeze_layers_cnn = freezeCnnLayersToKeys(self.params[TAG_BN])
                freeze_layers.update(freeze_layers_cnn)

            self.params = flax.core.frozen_dict.freeze(self.params)

        # print(freeze_layers)
        # print(jax.tree_map(jnp.shape, self.params))

        self.branch_apply = module_bn[0].apply
        self.trunk_apply = module_tn[0].apply

        self.opt_scheduler = exponential_decay(
            lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            step_offset=self.step_offset,
        )

        def optimizerSelector(path, _):
            if path in freeze_layers:
                return "none"
            elif path == TAG_ADAPTIVE:
                return "opt_adaptive_weights"
            else:
                return "opt"

        # path consists of ('params', 'tag', 'bias'), where 'tag' is the layer tag to freeze
        # 'b0' and 'adaptive_weights' are a special cases, where path = ('b0')
        tags = (
            [TAG_BN, TAG_TN, TAG_B0, TAG_ADAPTIVE]
            if len(adaptive_weights_shape) > 0
            else [TAG_BN, TAG_TN, TAG_B0]
        )
        label_fn = flattened_traversal(optimizerSelector, tags)

        self.optimizer = optax.chain(
            optax.clip(0.1),  # per-element clipping at 0.1 per paper (not global norm)
            optax.multi_transform(
                {
                    "opt": optax.adamw(learning_rate=self.opt_scheduler),
                    "opt_adaptive_weights": optax.adam(1e-5),  # hardcoded
                    "none": optax.set_to_zero(),
                },
                label_fn,
            ),
        )

        self.opt_state = self.optimizer.init(self.params)

    def train(
        self,
        dataloader,
        dataloader_val,
        nIter,
        save_every=200,
        do_timings=False,
        progress_callback: Callable[[float], None] | None = None,
    ):
        """Main train loop using dataloaders.

        Runs nIter additional iterations starting from self.step_offset, so when resuming
        from a checkpoint training continues from the last step and ends at step_offset + nIter.

        progress_callback, if given, is called after each iteration with a float in [0, 100]
        """
        writer = SummaryWriter(log_dir=self.log_dir)
        timer = TimingsWriter(log_dir=self.log_dir) if do_timings else None

        num_batches = np.ceil(dataloader.dataset.N / dataloader.batch_size)

        pbar_epochs = trange(np.ceil(nIter / num_batches).astype("int"))

        i = self.step_offset
        start = i
        last_pct = -1
        if i == 0:
            self.writeState(i, pbar_epochs, dataloader, dataloader_val, writer)
        if progress_callback is not None:
            progress_callback(0.0)
            last_pct = 0

        timer.resetTimings() if do_timings else None
        timer.startTiming("total_iter") if do_timings else None
        timer.startTiming("dataloader") if do_timings else None
        for _ in pbar_epochs:
            for _, data_batch in enumerate(dataloader):
                jax.block_until_ready(data_batch) if do_timings else None
                timer.endTiming("dataloader") if do_timings else None

                i += 1
                self._current_step = i

                if do_timings:
                    timer.startTiming("backprop")
                    self.params, self.opt_state, _ = self.step(
                        self.params, self.opt_state, data_batch
                    )
                    jax.block_until_ready(self.params)
                    jax.block_until_ready(self.opt_state)
                    timer.endTiming("backprop")

                    timer.writeTimings(
                        {
                            "total_iter": "Total time iter:",
                            "dataloader": "Dataloader:",
                            "backprop": "Back-propagation:",
                        }
                    )
                    timer.resetTimings()
                    timer.startTiming("total_iter")
                    timer.startTiming("dataloader")
                else:
                    self.params, self.opt_state, _ = self.step(
                        self.params, self.opt_state, data_batch
                    )

                if i % 100 == 0:
                    import sys
                    print(f"[heartbeat] step {i}/{start + nIter}", flush=True, file=sys.stderr)

                if i % save_every == 0:
                    self.writeState(i, pbar_epochs, dataloader, dataloader_val, writer)

                if progress_callback is not None:
                    pct = int(min(100.0, 100.0 * (i - start) / nIter))
                    if pct > last_pct:
                        progress_callback(float(pct))
                        last_pct = pct

        # save final result
        if i % save_every != 0:
            self.writeState(i, pbar_epochs, dataloader, dataloader_val, writer)

    def operator_net(self, params, B, y):
        trunk_params, b0 = params[TAG_TN], params[TAG_B0]
        T = self.trunk_apply(trunk_params, y)

        return jnp.sum(B * T) + b0

    def branch_net(self, params, u):
        branch_params = params[TAG_BN]
        if self.is_bn_fnn:
            return self.branch_apply(branch_params, u)
        else:
            return self.branch_apply(
                branch_params, expandCnnData(u), mutable=["batch_stats"]
            )[0]

    # Define total loss
    def loss(self, params, batch):
        return loss_functions.loss(
            params,
            batch,
            self.branch_net,
            self.operator_net,
            apply_adaptive_weights=TAG_ADAPTIVE in params,
        )

    # Define a compiled update step
    @partial(jit, static_argnums=(0))
    def step(self, params_all, opt_state, data_batch):
        def traverse_dict(fn, params):
            flat_dict = flax.traverse_util.flatten_dict(params)
            return flax.traverse_util.unflatten_dict(
                {k: fn(k, v) for k, v in flat_dict.items()}
            )

        idx_coord = data_batch[
            2
        ].flatten()  # get coordinate indexes for the current batch

        # extract adaptive weights for the current batch
        params = traverse_dict(
            lambda k, v: v[idx_coord] if TAG_ADAPTIVE in k else v, params_all
        )
        params = flax.core.frozen_dict.freeze(params)

        # calculate gradients for network parameters and adaptive weights
        loss_value, grads = jax.value_and_grad(self.loss)(params, data_batch)

        # negate gradient for adaptive weights
        grads = traverse_dict(lambda k, v: -v if TAG_ADAPTIVE in k else v, grads)
        grads = flax.core.frozen_dict.freeze(grads)

        # Scatter batch-subset adaptive_weights grads into a full-size zero array
        # so optax sees matching shapes between grads, opt_state, and params_all.
        # Without this, opt_state tracks (dataset.P,) but grads are (batch_size,).
        # Uses .add() (not .set()) so duplicate coordinate indices across batched
        # samples accumulate gradients additively — correct for same-coordinate
        # contributions from different source positions.
        if TAG_ADAPTIVE in grads:
            grad_aw_full = jnp.zeros_like(params_all[TAG_ADAPTIVE])
            grad_aw_full = grad_aw_full.at[idx_coord].add(grads[TAG_ADAPTIVE])
            grads = flax.core.frozen_dict.freeze({
                **{k: v for k, v in grads.items() if k != TAG_ADAPTIVE},
                TAG_ADAPTIVE: grad_aw_full,
            })

        # update ALL parameters (full adaptive_weights array, not just batch subset)
        # Wrap params_all in FrozenDict to match the optimizer's mask tree structure,
        # then unfreeze for apply_updates which expects plain dicts.
        params_all_frozen = flax.core.frozen_dict.freeze(params_all)
        updates, opt_state = self.optimizer.update(grads, opt_state, params_all_frozen)
        params_all_unfrozen = flax.core.frozen_dict.unfreeze(params_all_frozen)
        updates_unfrozen = flax.core.frozen_dict.unfreeze(updates)
        params = optax.apply_updates(params_all_unfrozen, updates_unfrozen)

        # clip adaptive_weights to [0, 1000] per paper
        if TAG_ADAPTIVE in params:
            params = flax.core.frozen_dict.freeze({
                **{k: v for k, v in params.items() if k != TAG_ADAPTIVE},
                TAG_ADAPTIVE: jnp.clip(params[TAG_ADAPTIVE], 0, 1000),
            })

        return params, opt_state, loss_value

    def plotLosses(self, figs_dir=None):
        plt.figure(figsize=(6, 5))
        plt.plot(
            self.loss_logger.nIter,
            self.loss_logger.loss_train,
            lw=2,
            label="Training loss",
        )
        plt.plot(
            self.loss_logger.nIter,
            self.loss_logger.loss_val,
            lw=2,
            label="Validation loss",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        if figs_dir is None:
            plt.show()
        else:
            fig_path = os.path.join(figs_dir, "loss.png")
            plt.savefig(fig_path, bbox_inches="tight", pad_inches=0)

    def writeState(self, it, pbar_epochs, dataloader_train, dataloader_val, writer):
        self.loss_logger.nIter.append(it)

        # training loss
        data_train_batch = next(iter(dataloader_train))
        loss_train_value = loss_functions.loss(
            self.params,
            data_train_batch,
            self.branch_net,
            self.operator_net,
            apply_adaptive_weights=False,
        )

        # validation loss
        data_val_batch = next(iter(dataloader_val))
        loss_val_value = loss_functions.loss(
            self.params,
            data_val_batch,
            self.branch_net,
            self.operator_net,
            apply_adaptive_weights=False,
        )

        # Store losses
        self.loss_logger.loss_train.append(loss_train_value)
        self.loss_logger.loss_val.append(loss_val_value)

        # Print losses
        pbar_epochs.set_postfix(
            {"Train loss": loss_train_value, "Val loss": loss_val_value}
        )

        # Save loss to disk
        self.writeSummary(writer, loss_train_value, loss_val_value, it)
        self.writeExperimentEvidence(it, loss_train_value, loss_val_value, dataloader_val)

        # Save model to disk
        self.writeModel(it)  # , write_separate=True
        self.writeTrainingCheckpoint(it)

    def writeModel(self, iter):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoints.save_checkpoint(
            ckpt_dir=self.log_dir,
            target=self.params,
            step=iter,
            overwrite=False,
            orbax_checkpointer=orbax_checkpointer,
        )

    def writeTrainingCheckpoint(self, iteration):
        if self.checkpoint_dir is None:
            return

        checkpoint_path = self.checkpoint_dir / f"epoch_{iteration:04d}"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpointer.save(
            checkpoint_path.resolve(),
            {
                "params": self.params,
                "opt_state": self.opt_state,
                "step": jnp.asarray(iteration, dtype=jnp.int32),
            },
            force=True,
        )
        metadata = {
            **self.checkpoint_metadata,
            "checkpoint_format": "orbax_pytree",
            "step": int(iteration),
            "state_keys": ["params", "opt_state", "step"],
        }
        (checkpoint_path / "checkpoint_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )


    def writeExperimentEvidence(self, it, loss_train, loss_val, dataloader_val):
        try:
            # 1. Calculate metrics
            batch = next(iter(dataloader_val))
            inputs, targets, _, _ = batch
            branch_input, trunk_input = inputs
            
            # predict_s(params, branch_input, trunk_input)
            # Use first sample in batch for quick validation
            pred = np.asarray(self.predict_s(self.params, branch_input[0], trunk_input[0]))
            tgt = np.asarray(targets[0]).flatten()
            
            mse = float(np.mean((pred.flatten() - tgt) ** 2))
            zero_rmse = float(np.sqrt(np.mean(tgt**2)))
            rel_rmse = float(np.sqrt(mse)) / zero_rmse if zero_rmse > 0 else 1.0
            improvement_db = float(10 * np.log10(zero_rmse**2 / mse)) if mse > 0 else 0.0
            
            metrics_path = Path(self.log_dir) / "metrics.jsonl"
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "step": int(it),
                    "train_loss": float(loss_train),
                    "val_loss": float(loss_val),
                    "rel_rmse": round(rel_rmse, 4),
                    "db_over_zero": round(improvement_db, 2),
                    "timestamp": time.time()
                }) + "\n")
                
            # 2. Plot loss curve
            plt.figure(figsize=(10, 6))
            plt.plot(self.loss_logger.nIter, self.loss_logger.loss_train, label="Train")
            plt.plot(self.loss_logger.nIter, self.loss_logger.loss_val, label="Val")
            plt.yscale("log")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"DeepONet Loss Curve (Step {it})")
            plots_dir = Path(self.log_dir) / "plots"
            plots_dir.mkdir(exist_ok=True)
            plt.savefig(plots_dir / f"loss_curve_{it:06d}.png")
            plt.close()
            
            # 3. Audio snippet
            audio_dir = Path(self.log_dir) / "audio"
            audio_dir.mkdir(exist_ok=True)
            min_len = min(len(pred.flatten()), len(tgt))
            stereo = np.stack([pred.flatten()[:min_len], tgt[:min_len]], axis=-1)
            # Use 2005 Hz as found in simulation_parameters.json
            sf.write(audio_dir / f"pred_vs_gt_{it:06d}.wav", stereo, 2005)
            
            print(f"[tracking] Step {it}: rel_rmse={rel_rmse:.4f}, improvement={improvement_db:.2f}dB")
        except Exception as e:
            print(f"[tracking] Failed to write evidence at step {it}: {e}")

    def writeSummary(self, writer, loss_train, loss_val, iter):
        writer.add_scalar("Loss/train/loss", np.array(loss_train), iter)
        writer.add_scalar("Loss/val/loss", np.array(loss_val), iter)
        writer.add_scalar(
            "Loss/learning_rate",
            np.array(self.opt_scheduler(iter - self.step_offset)),
            iter,
        )

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        branch_latent = self.branch_net(params, U_star)
        s_pred = vmap(self.operator_net, (None, None, 0))(params, branch_latent, Y_star)
        return s_pred


def loadLosses(path: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_acc = EventAccumulator(path)
    event_acc.Reload()

    scalar_tags = set(event_acc.Tags().get("scalars", []))

    def scalar_events(tag):
        return event_acc.Scalars(tag) if tag in scalar_tags else []

    learning_rate_events = scalar_events("Loss/learning_rate")
    train_events = scalar_events("Loss/train/loss")
    val_events = scalar_events("Loss/val/loss")
    available_events = learning_rate_events or train_events or val_events

    step_offset = available_events[-1].step if available_events else 0
    nIter = [event.step for event in train_events]
    loss_train = [event.value for event in train_events]
    loss_val = [event.value for event in val_events]

    return step_offset, LossLogger(loss_train, loss_val, nIter)
