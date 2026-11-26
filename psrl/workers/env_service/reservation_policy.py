"""Resolve a session reservation from an explicit execution capability contract."""

from dataclasses import dataclass

from .reuse_prediction import ReuseForecast

GPU_CONTRACTS = frozenset({"unknown", "independent_commands", "resident_state"})


@dataclass(frozen=True)
class ReservationDecision:
    mode: str
    reason: str


def choose_reservation(
    configured_mode: str,
    contract: str,
    needs_gpu: bool,
    forecast: ReuseForecast | None = None,
    *,
    require_timing: bool = False,
) -> ReservationDecision:
    """Resolve once before admission without inferring recoverability from utilization."""
    if contract not in GPU_CONTRACTS:
        raise ValueError(f"Unknown GPU execution contract {contract!r}.")
    if configured_mode not in ("episode", "command", "auto"):
        raise ValueError(f"Unknown reservation policy {configured_mode!r}.")
    if type(require_timing) is not bool:
        raise ValueError("Timing requirements must be boolean.")
    if not needs_gpu:
        return ReservationDecision("episode", "No GPU allocation is required.")
    if configured_mode != "auto":
        return ReservationDecision(configured_mode, "Explicit unit policy.")
    if contract == "independent_commands":
        if require_timing and forecast is None:
            return ReservationDecision("episode", "Cost-aware selection requires a calibrated timing forecast.")
        if forecast is not None:
            if not isinstance(forecast, ReuseForecast):
                raise ValueError("Timing advice requires a ReuseForecast.")
            return ReservationDecision(
                "command" if forecast.release_recommended else "episode",
                f"Timing profile with {forecast.samples} samples: {forecast.reason}",
            )
        return ReservationDecision("command", "Adapter declares independently returning managed GPU commands.")
    if contract == "resident_state":
        return ReservationDecision("episode", "Persistent GPU state requires a session reservation.")
    return ReservationDecision("episode", "Unknown execution capability retains a session reservation.")
