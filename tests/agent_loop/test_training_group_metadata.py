"""Keep single and grouped rollout identities valid across the training payload boundary."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from psrl.trainer.ppo.utils import compute_advantage_for_multi_trajectories
from psrl.workers.agent_loop.worker import PSRL_AgentLoopWorker
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.utils import tensordict_utils as tu

Worker = PSRL_AgentLoopWorker.__ray_metadata__.modified_class


@pytest.mark.parametrize("parent_id", [None, 3])
def test_output_payload_preserves_prompt_group_for_advantage(parent_id):
    worker = object.__new__(Worker)
    worker.overlong_filtering = False
    worker._compute_multi_modal_inputs = lambda *args: {}
    worker._compute_position_ids = lambda input_ids, *args: torch.zeros_like(input_ids)
    batch = TensorDict({}, batch_size=[1])
    tu.assign_non_tensor_stack(batch, "uid", [7])
    if parent_id is not None:
        tu.assign_non_tensor_stack(batch, "parent_id", [parent_id])
    tensors = {"prompts": torch.tensor([1, 2]), "responses": torch.tensor([3, 4]), "response_mask": torch.ones(2)}
    output = SimpleNamespace(prompt_ids=[1, 2], response_ids=[3, 4], as_dict=lambda: dict(tensors))
    keys, fields = worker._build_output_fields([output], batch, uid=7, version_tag=0)
    expected = 7 if parent_id is None else parent_id
    assert fields[0]["parent_id"] == expected, "Every training row must carry its actual prompt group."
    data = DataProto.from_dict(
        tensors={"response_mask": torch.ones(1, 2), "token_level_rewards": torch.tensor([[0.0, 0.5]])},
        non_tensors={"uid": np.array([7]), "parent_id": np.array([fields[0]["parent_id"]])},
    )
    result = compute_advantage_for_multi_trajectories(data, keys, AdvantageEstimator.GRPO)
    assert torch.isfinite(result.batch["advantages"]).all(), "The committed row must reach valid GRPO advantages."
    assert result.batch["advantages"].shape == (1, 2), "Advantage selection must preserve the sample dimension."
