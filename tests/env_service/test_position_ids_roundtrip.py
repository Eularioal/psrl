"""Exercise stored multimodal position rows through training batch selection."""

import pytest
import torch
from tensordict import TensorDict
from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
from verl.utils.tensordict_utils import index_select_tensor_dict, maybe_fix_3d_position_ids


@pytest.mark.parametrize("lengths", [(2407,), (9, 13), (11, 11)])
def test_stored_position_ids_keep_coordinates_and_token_lengths(lengths):
    rows = [torch.arange(4 * size).reshape(4, size) for size in lengths]
    packed = AsyncSimpleStorageManager._pack_field_values(rows)
    batch = TensorDict({"position_ids": packed}, batch_size=[len(rows)])
    maybe_fix_3d_position_ids(batch)
    maybe_fix_3d_position_ids(batch)
    selected = index_select_tensor_dict(batch, list(reversed(range(len(rows)))))
    actual = selected["position_ids"].unbind()
    assert all(torch.equal(left, right) for left, right in zip(actual, reversed(rows))), "Position rows changed."
    assert selected["position_ids"].offsets().diff().tolist() == list(reversed(lengths)), "Token lengths changed."
