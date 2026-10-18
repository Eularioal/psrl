"""Execute APPS inputs in a container that has no private answers or dataset mounts."""

import contextlib
import io
import json
import os
import selectors
import signal
import sys

OUTPUT_LIMIT = 1024 * 1024
PRELUDE = """
import sys, time, itertools, collections, math, fractions, random, heapq
from itertools import accumulate, product, permutations, combinations
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
from typing import List, Tuple
from heapq import *
import numpy as np
"""


def execute(compiled, inputs, function, common):
    """Fork for each input so stdin, globals and Python library mutations cannot persist."""
    text = "\n".join(inputs) if function is None and isinstance(inputs, list) else inputs
    input_fd = os.memfd_create("apps-input", 0)
    if function is None:
        os.write(input_fd, text.encode())
        os.lseek(input_fd, 0, os.SEEK_SET)
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(out_r)
            os.close(err_r)
            os.dup2(input_fd, 0)
            os.dup2(out_w, 1)
            os.dup2(err_w, 2)
            os.close(input_fd)
            os.close(out_w)
            os.close(err_w)
            sys.stdin = io.TextIOWrapper(os.fdopen(0, "rb", closefd=False))
            sys.stdout = io.TextIOWrapper(os.fdopen(1, "wb", closefd=False), write_through=True)
            sys.stderr = io.TextIOWrapper(os.fdopen(2, "wb", closefd=False), write_through=True)
            namespace = {**common, "__name__": "__main__"}
            if function is None:
                try:
                    exec(compiled, namespace)
                except SystemExit as error:
                    if error.code not in (None, 0):
                        raise
            else:
                if not isinstance(inputs, list):
                    raise ValueError("Function inputs must be an argument list.")
                if inputs and isinstance(inputs[0], dict):
                    try:
                        inputs[0] = {int(k): v for k, v in inputs[0].items()}
                    except (ValueError, TypeError, OverflowError):
                        pass
                with contextlib.redirect_stdout(sys.stderr):
                    exec(compiled, namespace)
                    owner = namespace["Solution"]() if "Solution" in namespace else None
                    fn = getattr(owner, function) if owner is not None else namespace[function]
                    result = fn(*inputs)

                def serialize_numpy(value):
                    if isinstance(value, common["np"].ndarray):
                        return value.tolist()
                    if isinstance(value, common["np"].generic):
                        return value.item()
                    raise TypeError("Function returned a value outside the JSON contract.")

                print(json.dumps(result, allow_nan=False, default=serialize_numpy))
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        except BaseException:
            os._exit(1)
    os.close(input_fd)
    os.close(out_w)
    os.close(err_w)
    chunks, size = [], 0
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(out_r, selectors.EVENT_READ)
            selector.register(err_r, selectors.EVENT_READ)
            while selector.get_map():
                for key, _ in selector.select():
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fd)
                        continue
                    size += len(data)
                    if size > OUTPUT_LIMIT:
                        raise ValueError("Candidate output exceeded its limit.")
                    if key.fd == out_r:
                        chunks.append(data)
        _, status = os.waitpid(pid, 0)
        pid = None
        return {"ok": os.waitstatus_to_exitcode(status) == 0, "stdout": b"".join(chunks).decode(errors="replace")}
    except Exception:
        return {"ok": False, "stdout": ""}
    finally:
        if pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        os.close(out_r)
        os.close(err_r)


def main():
    """Receive source once and one test input at a time over an untrusted data channel."""
    common = {}
    exec(PRELUDE, common)
    channel = sys.stdin.buffer
    print(json.dumps({"ready": True}), flush=True)
    config = json.loads(channel.readline(2 * 1024 * 1024))
    try:
        compiled = compile(config["code"], "submission.py", "exec")
    except (SyntaxError, ValueError):
        compiled = None
    for line in channel:
        request = json.loads(line)
        result = {"ok": False, "stdout": ""}
        if compiled is not None:
            result = execute(compiled, request["input"], config["fn_name"], common)
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
