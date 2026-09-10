"""Test streaming rollout without a large checkpoint or GPU."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytest.importorskip("anemoi.inference.config.run")

from anemoi.inference.runners.predictive_autoencoder import PredictiveAutoencoderRunner
from anemoi.models.models.predictive_autoencoder import AnemoiModelPredictiveAutoEncoder


@pytest.mark.parametrize("previous_state", [False, True])
def test_streaming_rollout(previous_state):
    core = Mock(spec=AnemoiModelPredictiveAutoEncoder)
    core.use_previous_state = previous_state
    core.encode_snapshot.side_effect = lambda x, t, **kw: (x["data"][:, t:t+1, ..., :1].clone(), None)
    core.encode_static_forcing_context.return_value = (None, None)
    core.encode_forcing_context.side_effect = lambda x, t, **kw: (x["data"][..., 1:], None)
    core.transition_latent.side_effect = lambda previous, current, context, **kw: current + context
    core.decode_snapshot.side_effect = lambda latent, *a, **kw: {"data": latent}
    identity = lambda x, **kw: x
    model = SimpleNamespace(
        model=core, eval=lambda: None,
        pre_processors={"data": identity}, post_processors={"data": identity},
        data_indices={"data": SimpleNamespace(data=SimpleNamespace(output=SimpleNamespace(ordered_names=["p"])))},
    )
    start = datetime(2023, 1, 1)
    def forcing(tensor, state, dates, check):
        tensor[:, -1, :, 1] = (dates[0] - start) / timedelta(hours=6)
        return tensor
    runner = object.__new__(PredictiveAutoencoderRunner)
    runner._device = torch.device("cpu")
    runner.__dict__["model"] = model
    runner.__dict__["autocast"] = torch.bfloat16
    runner._checkpoint = SimpleNamespace(timestep=timedelta(hours=6))
    runner.tensor_handlers = {"data": SimpleNamespace(
        add_dynamic_forcings_to_input_tensor=forcing, boundary_forcings_providers=[],
    )}
    values = np.zeros((1 + int(previous_state), 2, 3), dtype=np.float32)
    outputs = list(runner.forecast("240h", {"data": values}, {"data": {"date": start, "fields": {}}}))
    assert len(outputs) == 40
    assert outputs[0]["data"]["date"] == start + timedelta(hours=6)
    assert outputs[-1]["data"]["step"] == timedelta(hours=240)
    assert core.encode_snapshot.call_count == 1 + int(previous_state)
    assert core.encode_static_forcing_context.call_count == 1
    assert core.transition_latent.call_count == core.decode_snapshot.call_count == 40
    torch.testing.assert_close(outputs[-1]["data"]["fields"]["p"], torch.full((3,), 820.0))
