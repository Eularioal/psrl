"""Check ownership and cancellation boundaries without a GPU."""

import asyncio

import pytest
from psrl.workers.env_service.gpu_execution import GpuExecutionBackend
from psrl.workers.env_service.mps import MpsClientController, parse_clients


@pytest.mark.parametrize("row", ["0 1 20 GPU-a ? ?", "10 1", "0 1 20 GPU-a 99 python"])
def test_incomplete_identity_rejected(row):
    with pytest.raises(RuntimeError):
        parse_clients(row)


@pytest.mark.parametrize("change", ["epoch", "birth", "server", "connection", "response"])
def test_stale_binding_cannot_release(change):
    identities = {10: (100, 99), 20: (200, 88)}
    table = ["10 1 20 GPU-a 99 python"]
    commands = []

    def control(command):
        commands.append(command)
        return table[0] if command == "ps" else "7" if change == "response" else "0"

    controller = MpsClientController(control, identities.__getitem__)
    binding = controller.bind("job", 1, 10)
    if change == "birth":
        identities[10] = (101, 99)
    elif change == "server":
        identities[20] = (201, 88)
    elif change == "connection":
        table[0] = "10 2 20 GPU-a 99 python"
    with pytest.raises(RuntimeError):
        controller.terminate(binding, "job", 2 if change == "epoch" else 1)
    if change != "response":
        assert not any(command.startswith("terminate") for command in commands)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "mps", "process"])
async def test_cancel_order_and_failed_cleanup_retains_capacity(failure):
    events = []

    class Policy:
        mode = "episode"

        async def acquire(self, *args):
            events.append("acquire")

        async def cancel(self, record):
            events.append("release")

    async def owner():
        return "container"

    async def run():
        await asyncio.sleep(10)

    async def mps():
        events.append("mps")
        if failure == "mps":
            raise RuntimeError("MPS failure.")

    async def process():
        events.append("process")
        if failure == "process":
            raise RuntimeError("Process failure.")

    with pytest.raises(RuntimeError if failure else TimeoutError):
        await GpuExecutionBackend(Policy()).execute(
            run, owner, timeout_sec=0.01, wait_timeout_sec=1, before_cancel=mps, cancel=process
        )
    expected = {
        None: ["acquire", "mps", "process", "release"],
        "mps": ["acquire", "mps"],
        "process": ["acquire", "mps", "process"],
    }
    assert events == expected[failure]
