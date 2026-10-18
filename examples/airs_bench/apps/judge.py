"""Keep private APPS answers in a trusted process and request isolated execution."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparison import matches  # noqa: E402

MAX_WIRE_BYTES = 2 * 1024 * 1024


class InvalidSubmission(ValueError):
    """The submitted CSV violates the public submission contract."""


def request(message):
    """Exchange bounded JSON with the worker, never with a candidate process directly."""
    try:
        wire = json.dumps(message, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise RuntimeError("APPS judge request is not valid protocol data.") from error
    if len(wire.encode()) >= MAX_WIRE_BYTES:
        raise RuntimeError("APPS input exceeds the protocol limit.")
    print(wire, flush=True)
    line = sys.stdin.buffer.readline(MAX_WIRE_BYTES)
    if not line.endswith(b"\n"):
        raise RuntimeError("Judge control channel was closed or exceeded its limit.")
    return json.loads(line)


def load_tests(sample):
    """Reject malformed private fixtures as infrastructure failures."""
    try:
        tests = json.loads(sample["input_output"])
        inputs, expected = tests["inputs"], tests["outputs"]
        function = tests.get("fn_name")
        if (
            not isinstance(inputs, list)
            or not isinstance(expected, list)
            or not inputs
            or len(inputs) != len(expected)
        ):
            raise ValueError("Mismatched test cases.")
        if function is not None and (not isinstance(function, str) or not function):
            raise ValueError("Invalid function name.")
        if function is not None:
            if not all(isinstance(value, list) for value in inputs):
                raise ValueError("Function inputs must be argument lists.")
        elif not all(
            isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, str) for item in value))
            for value in [*inputs, *expected]
        ):
            raise ValueError("Standard input cases must contain text.")
        return inputs, expected, function
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise RuntimeError("Private APPS cases are malformed.") from error


def grade(rows, samples):
    """Count a problem only when one of its five programs passes every case."""
    if not samples:
        raise RuntimeError("Private APPS split is empty.")
    if len(rows) != len(samples):
        raise InvalidSubmission("APPS submission row count must match the test split.")
    if any(len(row) != 5 for row in rows):
        raise InvalidSubmission("Every APPS row requires five candidates.")
    if any(not isinstance(source, str) or len(source.encode()) > 256 * 1024 for row in rows for source in row):
        raise InvalidSubmission("APPS source must fit within 256 KiB per candidate.")
    correct = 0
    for row, sample in zip(rows, samples, strict=True):
        inputs, expected, function = load_tests(sample)
        for source in row:
            request({"op": "begin", "code": source, "fn_name": function})
            passed = True
            try:
                for inputs_one, expected_one in zip(inputs, expected, strict=True):
                    response = request({"op": "case", "input": inputs_one})
                    if not response["ok"]:
                        passed = False
                        break
                    actual = response["stdout"]
                    if function is not None:
                        try:
                            actual = json.loads(actual)
                        except (ValueError, TypeError):
                            passed = False
                            break
                    if not matches(actual, expected_one, function is not None):
                        passed = False
                        break
            finally:
                request({"op": "end"})
            if passed:
                correct += 1
                break
    return {"Pass@5": correct / len(samples)}


def main():
    """Load only trusted dataset files and treat submission cells as opaque text."""
    from datasets import load_from_disk

    csv.field_size_limit(256 * 1024)
    try:
        with open("/home/agent/workspace/submission.csv", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != [f"code{i}" for i in range(1, 6)]:
                raise ValueError("APPS CSV requires exactly code1 through code5.")
            rows = list(reader)
        if any(len(row) != 5 for row in rows):
            raise ValueError("Every APPS row requires five candidates.")
    except (ValueError, csv.Error):
        print(json.dumps({"op": "invalid_submission"}), flush=True)
        return
    samples = load_from_disk("/home/agent/workspace/data/test_with_labels")
    try:
        metrics = grade(rows, samples)
    except InvalidSubmission:
        print(json.dumps({"op": "invalid_submission"}), flush=True)
        return
    print(json.dumps({"op": "result", "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
