"""
Execute commands through a persistent sandbox shell.

The completion sentinel preserves shell state while exposing each command's exit status.
"""

from __future__ import annotations

import logging
import os
import re

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

PROCESS_DONE_MARKER_START = "///PROCESS-DONE:"
PROCESS_DONE_MARKER_END = ":PROCESS-DONE///"

TRUNCATION_NOTICE = "\n... {omitted} characters omitted ...\n"


def wrap_command(command: str, command_id: str) -> str:
    """
    Append the completion sentinel to a shell command.

    Each invocation carries a unique id so unrelated markers cannot finish it.

    Args:
        command (str): Raw shell command from the agent.

    Returns:
        str: Command text ready to write to the shell's stdin.
    """
    terminated = command if command.endswith("\n") else command + "\n"
    sentinel_echo = (
        f'EXITSTATUS="$?"; printf "\\n%s\\n" "{PROCESS_DONE_MARKER_START}{command_id}:$EXITSTATUS'
        f'{PROCESS_DONE_MARKER_END}"\n'
    )
    return terminated + sentinel_echo


def parse_sentinel(buffer: str, command_id: str) -> tuple[str, int] | None:
    """
    Split a shell output buffer at the completion sentinel.

    Args:
        buffer (str): Accumulated shell output.

    Returns:
        tuple[str, int] | None: Output and integer status for the requested command,
            or None when its complete marker has not arrived.
    """
    pattern = re.compile(
        rf"{re.escape(PROCESS_DONE_MARKER_START + command_id)}:(\d+){re.escape(PROCESS_DONE_MARKER_END)}"
    )
    match = pattern.search(buffer)
    if match is None:
        return None

    exit_code = int(match.group(1))
    body = buffer[: match.start()]
    return body, exit_code


def truncate_observation(text: str, max_chars: int) -> str:
    """
    Cap an observation to a character budget, keeping the head and the tail.

    MLGym does not bound command output, so a single training run can emit enough
    text to exhaust the model context in one turn. The head carries the command
    echo and early errors, the tail carries the final result.

    The total output length is at most `max_chars * 2`: the notice is accounted for
    inside the budget so the caller can rely on this bound.

    Args:
        text (str): Raw observation text.
        max_chars (int): Total character budget split evenly between head and tail.
            The elision notice is deducted from the budget before splitting.
            Values of 0 or less disable truncation.

    Returns:
        str: Original text, or a head plus notice plus tail whose combined length
            does not exceed `max_chars * 2`.
    """
    if max_chars <= 0 or len(text) <= max_chars * 2:
        return text

    # The notice length stabilizes once the omitted digit count is known.
    total_budget = max_chars * 2
    placeholder_notice = TRUNCATION_NOTICE.format(omitted=len(text))
    per_side = (total_budget - len(placeholder_notice)) // 2
    if per_side <= 0:
        per_side = 0

    omitted = len(text) - per_side * 2
    notice = TRUNCATION_NOTICE.format(omitted=omitted)
    psrl_logger.debug(f"Truncated a sandbox observation. Omitted characters={omitted}.")
    head = text[:per_side] if per_side > 0 else ""
    tail = text[-per_side:] if per_side > 0 else ""
    return head + notice + tail
