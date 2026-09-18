"""Streaming latent forecasts using Anemoi's standard inputs and outputs."""

from copy import deepcopy
from functools import cached_property

import numpy as np
import torch
from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.runner import RunnerClasses
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.models.models.predictive_autoencoder import AnemoiModelPredictiveAutoEncoder

from . import runner_registry
from .default import DefaultRunner


class PredictiveCheckpoint(Checkpoint):
    """Present the codec's analysis inputs and one forecast output to inference."""

    @cached_property
    def _raw_metadata(self):
        metadata, arrays = super()._raw_metadata
        metadata = deepcopy(metadata)
        inference = metadata["metadata_inference"]
        previous = metadata["config"]["task"].get("use_previous_state", False)
        for name in inference["dataset_names"]:
            timing = inference[name]["timesteps"]
            timing["input_relative_date_indices"] = [-1, 0] if previous else [0]
            timing["output_relative_date_indices"] = [1]
            for key in ("input_offsets", "output_offsets", "rollout_shift", "advance_map"):
                timing.pop(key, None)
        return metadata, arrays


@runner_registry.register("predictive-autoencoder")
class PredictiveAutoencoderRunner(DefaultRunner):
    """Encode initial conditions once and retain the latent state during rollout."""

    def __init__(self, config):
        super().__init__(config, classes=RunnerClasses(checkpoint=PredictiveCheckpoint))
        if self.mid_processors and any(self.mid_processors.values()):
            raise ValueError("Mid-processors cannot update a persistent latent state.")

    def forecast(self, lead_time, input_tensors_numpy, input_states):
        model = self.model
        core = model.model
        if not isinstance(core, AnemoiModelPredictiveAutoEncoder):
            raise TypeError("This runner requires a predictive-autoencoder inference checkpoint.")
        horizon = frequency_to_timedelta(lead_time)
        timestep = self.checkpoint.timestep
        steps = horizon / timestep
        if steps < 0 or not float(steps).is_integer():
            raise ValueError("lead_time must be a non-negative multiple of the model timestep.")
        starts = {state["date"] for state in input_states.values()}
        if len(starts) != 1:
            raise ValueError("All initial states must have the same date.")
        start = starts.pop()
        model.eval()
        tensors = {
            name: torch.as_tensor(np.swapaxes(value, -2, -1)[None], device=self.device)
            for name, value in input_tensors_numpy.items()
        }

        def preprocess(values):
            return {
                name: model.pre_processors[name](value[:, :, None], in_place=False)
                for name, value in values.items()
            }

        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.autocast):
            initial = preprocess(tensors)
            previous = None
            if core.use_previous_state:
                previous, _ = core.encode_snapshot(initial, 0, batch_size=1)
            current, shards = core.encode_snapshot(initial, int(core.use_previous_state), batch_size=1)
            static, _ = core.encode_static_forcing_context(initial, batch_size=1)
            del initial
            states = {name: {**state, "step": timestep * 0} for name, state in input_states.items()}
            for step in range(1, int(steps) + 1):
                date = start + step * timestep
                for name, handler in self.tensor_handlers.items():
                    check = np.zeros(tensors[name].shape[-1], dtype=bool)
                    tensors[name] = handler.add_dynamic_forcings_to_input_tensor(
                        tensors[name], states[name], [date], check
                    )
                    if handler.boundary_forcings_providers:
                        raise ValueError("Boundary updates require a latent assimilation mechanism.")
                target = preprocess({name: value[:, -1:] for name, value in tensors.items()})
                context, _ = core.encode_forcing_context(target, 0, batch_size=1, static_context=static)
                predicted = core.transition_latent(previous, current, context, batch_size=1, shard_sizes_hidden=shards)
                decoded = core.decode_snapshot(
                    predicted, target, 0, batch_size=1, ensemble_size=1,
                    shard_sizes_hidden=shards, in_out_sharded={name: False for name in tensors},
                )
                previous, current = current, predicted
                for name, value in decoded.items():
                    physical = model.post_processors[name](value, in_place=False)[0, 0, 0].float()
                    names = model.data_indices[name].data.output.ordered_names
                    states[name] = {
                        **states[name], "date": date, "step": step * timestep,
                        "previous_step": (step - 1) * timestep,
                        "fields": {variable: physical[:, i] for i, variable in enumerate(names)},
                    }
                yield dict(states)
                del decoded, target, context
