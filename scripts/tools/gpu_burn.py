# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep eight local GPUs busy with repeated matrix multiplication.

Run from the repository root with a CUDA-enabled PyTorch installation::

    python -m scripts.tools.gpu_burn
    python -m scripts.tools.gpu_burn --duration 60

CUDA_VISIBLE_DEVICES controls device visibility. Ctrl+C or SIGTERM sent to the
parent stops all workers. No Ray cluster or model files are required.
"""

import argparse
import math
import multiprocessing as mp
import signal
import time
from multiprocessing.synchronize import Event
from types import FrameType
from typing import Optional

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def burn_gpu(device: int, args: argparse.Namespace, stop: Event) -> None:
    """Run bounded batches of GEMMs, polling completion without a GPU sync."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import torch

    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"GPU {device} does not support bfloat16; use --dtype float16")

    with torch.inference_mode():
        shape = (args.matrix_size, args.matrix_size)
        left = torch.empty(shape, device=f"cuda:{device}", dtype=dtype).normal_(std=0.01)
        right = torch.empty_like(left).normal_(std=0.01)
        output = torch.empty_like(left)
        completed = torch.cuda.Event()
        logger.info(
            "GPU %d (%s): starting %s GEMMs, size=%d",
            device,
            torch.cuda.get_device_name(),
            args.dtype,
            args.matrix_size,
        )
        while not stop.is_set():
            for _ in range(args.batch_size):
                if stop.is_set():
                    return
                torch.mm(left, right, out=output)
            completed.record()
            # Bound the asynchronous work queue and keep shutdown responsive.
            while not completed.query():
                if stop.wait(0.001):
                    return
            if args.sleep_ms and stop.wait(args.sleep_ms / 1000):
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of visible GPUs to use (default: 8)")
    parser.add_argument("--matrix-size", type=int, default=8192, help="Square matrix dimension (default: 8192)")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--batch-size", type=int, default=32, help="GEMMs per batch (default: 32)")
    parser.add_argument("--duration", type=float, default=0, help="Seconds including worker startup; 0 runs forever")
    parser.add_argument(
        "--sleep-ms", type=float, default=0, help="Idle milliseconds between batches; 0 maximizes load"
    )
    args = parser.parse_args()
    for name in ("num_gpus", "matrix_size", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("duration", "sleep_ms"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    return args


def main() -> int:
    args = parse_args()

    import torch

    available = torch.cuda.device_count()
    if available < args.num_gpus:
        logger.error(
            "Requested %d GPUs, but only %d are visible. Check CUDA_VISIBLE_DEVICES.", args.num_gpus, available
        )
        return 1

    context = mp.get_context("spawn")
    stop = context.Event()
    workers = []
    interrupted = False

    def request_stop(signum: int, frame: Optional[FrameType]) -> None:
        # Avoid taking multiprocessing locks inside a signal handler.
        nonlocal interrupted
        interrupted = True

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    result = 0
    started = time.monotonic()
    try:
        logger.info(
            "Starting %d GPU workers. Stop with Ctrl+C or SIGTERM to parent PID %d.",
            args.num_gpus,
            mp.current_process().pid,
        )
        for device in range(args.num_gpus):
            if interrupted:
                break
            worker = context.Process(target=burn_gpu, args=(device, args, stop), name=f"gpu-burn-{device}")
            worker.start()
            workers.append(worker)
        while not interrupted:
            exited = [worker for worker in workers if worker.exitcode is not None]
            if exited:
                for worker in exited:
                    logger.error("%s exited unexpectedly with code %s", worker.name, worker.exitcode)
                result = 1
                break
            if args.duration and time.monotonic() - started >= args.duration:
                break
            time.sleep(0.2)
    finally:
        stop.set()
        deadline = time.monotonic() + 5
        for worker in workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        deadline = time.monotonic() + 3
        for worker in workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))
            if worker.is_alive():
                worker.kill()
                worker.join()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        logger.info("All GPU workers stopped.")
    return result


if __name__ == "__main__":
    raise SystemExit(main())

