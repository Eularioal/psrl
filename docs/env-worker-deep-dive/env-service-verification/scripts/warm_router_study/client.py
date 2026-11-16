"""Send one bounded request to the episode-owned GPU service."""

import argparse
import json
import socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(240)
        connection.connect(args.socket)
        connection.sendall((args.request + "\n").encode())
        result = json.loads(connection.makefile("rb").readline())
    print(json.dumps(result), flush=True)
    if result.get("status") != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
