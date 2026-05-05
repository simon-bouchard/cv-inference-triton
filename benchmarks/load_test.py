#!/usr/bin/env python3
"""
Pipeline load tester for Triton Inference Server.

Usage:
    python load_test.py --model yolov8s_pipeline --image data/sample.jpg
    python load_test.py --model yolov8s_pipeline --image data/sample.jpg --concurrency 4
    python load_test.py --model yolov8s_pipeline --image data/sample.jpg --concurrency 1:8
    python load_test.py --model yolov8s_pipeline --image data/sample.jpg --protocol grpc

Concurrency range format: single value or start:end (e.g. 1:8 sweeps 1,2,3,4,5,6,7,8)
Output is printed to stdout and optionally saved to a CSV file.
"""

import argparse
import csv
import threading
import time
from pathlib import Path

import numpy as np

# ── argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Triton model name")
parser.add_argument("--image", required=True, help="Path to input image")
parser.add_argument("--url", default="localhost", help="Triton server host")
parser.add_argument("--protocol", default="http", choices=["http", "grpc"])
parser.add_argument("--concurrency", default="1", help="Single value or start:end")
parser.add_argument("--duration", default=30, type=int, help="Seconds per concurrency level")
parser.add_argument("--output", default=None, help="Path to output CSV file")
parser.add_argument("--outputs", default="boxes,scores,class_ids", help="Comma-separated output tensor names")
args = parser.parse_args()

# ── concurrency range ─────────────────────────────────────────────────────────

if ":" in args.concurrency:
    start, end = map(int, args.concurrency.split(":"))
    concurrency_levels = list(range(start, end + 1))
else:
    concurrency_levels = [int(args.concurrency)]

# ── load image ────────────────────────────────────────────────────────────────

with open(args.image, "rb") as f:
    image_bytes = f.read()

# ── client factory ────────────────────────────────────────────────────────────


def make_client():
    if args.protocol == "http":
        import tritonclient.http as client_lib

        client = client_lib.InferenceServerClient(url=f"{args.url}:8000")
    else:
        import tritonclient.grpc as client_lib

        client = client_lib.InferenceServerClient(url=f"{args.url}:8001")
    return client, client_lib


def make_inputs(client_lib):
    image_data = np.array([image_bytes], dtype=object)
    inp = client_lib.InferInput("image_raw", [1], "BYTES")
    inp.set_data_from_numpy(image_data)
    return [inp]


def make_outputs(client_lib):
    return [client_lib.InferRequestedOutput(name) for name in args.outputs.split(",")]


# ── worker ────────────────────────────────────────────────────────────────────


def worker(model, duration, latencies, errors, stop_event):
    client, client_lib = make_client()
    inputs = make_inputs(client_lib)
    outputs = make_outputs(client_lib)

    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            client.infer(model, inputs=inputs, outputs=outputs)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms
        except Exception as e:
            errors.append(str(e))


# ── benchmark one concurrency level ──────────────────────────────────────────


def run_level(concurrency):
    latencies = []
    errors = []
    stop_event = threading.Event()

    threads = [
        threading.Thread(
            target=worker, args=(args.model, args.duration, latencies, errors, stop_event)
        )
        for _ in range(concurrency)
    ]

    t_start = time.perf_counter()
    for t in threads:
        t.start()

    time.sleep(args.duration)
    stop_event.set()

    for t in threads:
        t.join()

    elapsed = time.perf_counter() - t_start

    if not latencies:
        print(f"  [c={concurrency}] No successful requests. Errors: {errors[:3]}")
        return None

    arr = np.array(latencies)
    throughput = len(arr) / elapsed
    p50 = np.percentile(arr, 50)
    p99 = np.percentile(arr, 99)

    return {
        "model": args.model,
        "protocol": args.protocol,
        "concurrency": concurrency,
        "requests": len(arr),
        "errors": len(errors),
        "p50_ms": round(p50, 2),
        "p99_ms": round(p99, 2),
        "throughput": round(throughput, 2),
    }


# ── main ──────────────────────────────────────────────────────────────────────

print(f"\nModel:    {args.model}")
print(f"Protocol: {args.protocol}")
print(f"Duration: {args.duration}s per level")
print(f"Concurrency levels: {concurrency_levels}\n")
print(f"{'Concurrency':>12}  {'p50 (ms)':>10}  {'p99 (ms)':>10}  {'Throughput':>12}  {'Errors':>8}")
print("-" * 60)

rows = []
for c in concurrency_levels:
    result = run_level(c)
    if result:
        rows.append(result)
        print(
            f"{result['concurrency']:>12}  {result['p50_ms']:>10}  {result['p99_ms']:>10}  {result['throughput']:>12}  {result['errors']:>8}"
        )

# ── csv output ────────────────────────────────────────────────────────────────

if args.output and rows:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {args.output}")
