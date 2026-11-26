"""Validate queue placement and acknowledged inference to environment transitions."""

import asyncio

import pytest
from psrl.workers.env_service.elastic_partition import ElasticPartition, PartitionDevice


def pool(dynamic=True, sleep=None, wake=None):
    async def noop(name):
        pass

    return ElasticPartition(
        [
            PartitionDevice("e0", 1, 10),
            PartitionDevice("e1", 1, 10),
            PartitionDevice("r0", 1, 10, True),
            PartitionDevice("r1", 1, 10, True),
        ],
        sleep or noop,
        wake or noop,
        dynamic=dynamic,
        transition_timeout_s=1,
    )


@pytest.mark.asyncio
async def test_resident_sharing_keeps_inference_active_and_dedicated_priority():
    controller = pool(False)
    controller.resident_sharing = True
    async with controller.environment(8) as first, controller.environment(8) as second:
        async with controller.inference() as model:
            async with controller.environment(8) as shared:
                assert (first, second) == ("e0", "e1"), "Dedicated devices must be preferred."
                assert shared == model == "r0", "Resident mode must allow explicit coexistence."
                assert controller.devices[2].state == "inference", "Resident sharing must retain inference routing."
                await controller.reconcile()
                assert not any(e["action"] == "drain_start" for e in controller.events), "No model may sleep."


@pytest.mark.asyncio
async def test_fixed_pool_uses_next_device_and_queues():
    controller = pool(False)
    async with controller.environment(8) as first, controller.environment(8) as second:
        assert (first, second) == ("e0", "e1")
        waiting = asyncio.create_task(controller.environment(8).__aenter__())
        await asyncio.sleep(0)
        await controller.reconcile()
        assert not waiting.done()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert not controller.waiters


@pytest.mark.asyncio
async def test_donation_waits_for_inference_and_restore_waits_for_environment():
    slept = asyncio.Event()
    woke = asyncio.Event()

    async def sleep(name):
        slept.set()

    async def wake(name):
        woke.set()

    controller = pool(sleep=sleep, wake=wake)
    release = asyncio.Event()
    started = asyncio.Event()

    async def command():
        async with controller.environment(8) as device:
            assert device.startswith("r")
            started.set()
            await release.wait()

    async with controller.environment(8), controller.environment(8):
        one = controller.inference()
        two = controller.inference()
        await one.__aenter__()
        await two.__aenter__()
        job = asyncio.create_task(command())
        await asyncio.sleep(0)
        transition = asyncio.create_task(controller.reconcile())
        await asyncio.sleep(0)
        assert not slept.is_set()
        await one.__aexit__(None, None, None)
        await transition
        await started.wait()
        restore = asyncio.create_task(controller.reconcile())
        await asyncio.sleep(0)
        assert not woke.is_set()
        release.set()
        await job
        await restore
        await two.__aexit__(None, None, None)
        assert woke.is_set()


@pytest.mark.asyncio
async def test_failed_sleep_is_never_advertised_as_free():
    async def fail(name):
        raise RuntimeError("Sleep failed.")

    controller = pool(sleep=fail)
    async with controller.environment(8), controller.environment(8):
        waiting = asyncio.create_task(controller.environment(8).__aenter__())
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="Sleep failed"):
            await controller.reconcile()
        assert controller.devices[2].state == "quarantined"
        assert not waiting.done()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting


@pytest.mark.asyncio
async def test_busy_inference_prevents_consolidation():
    controller = pool()
    contexts = [controller.inference() for _ in range(5)]
    for context in contexts:
        await context.__aenter__()
    async with controller.environment(8), controller.environment(8):
        waiting = asyncio.create_task(controller.environment(8).__aenter__())
        await asyncio.sleep(0)
        await controller.reconcile()
        assert all(d.state == "inference" for d in controller.devices if d.inference)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
    for context in contexts:
        await context.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_cancelled_rpc_keeps_inference_device_quarantined():
    controller = pool()
    with pytest.raises(asyncio.CancelledError):
        async with controller.inference():
            raise asyncio.CancelledError
    assert controller.devices[2].state == "quarantined"
    assert not controller.devices[2].infer_active
