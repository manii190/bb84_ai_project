"""
Benchmark the exact batched BB84 simulator using multiple CPU cores.

This keeps:
- exact photon-by-photon BB84 simulation
- Eve intercept-resend
- photon loss
- channel noise
- all seven machine-learning features
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from bb84_batch_simulator import generate_exact_bb84_batch


def generate_worker(arguments):
    """
    Generate one independent BB84 batch.

    This function must remain outside other functions so that
    multiprocessing works correctly on Windows.
    """
    batch_size, number_of_bits, seed = arguments

    return generate_exact_bb84_batch(
        number_of_sessions=batch_size,
        number_of_bits=number_of_bits,
        seed=seed
    )


def create_batch_jobs(
    number_of_sessions,
    batch_size,
    number_of_bits,
    starting_seed
):
    """
    Divide the requested dataset into independent batch jobs.
    """
    jobs = []
    generated = 0
    batch_number = 0

    while generated < number_of_sessions:
        current_batch_size = min(
            batch_size,
            number_of_sessions - generated
        )

        current_seed = starting_seed + batch_number

        jobs.append(
            (
                current_batch_size,
                number_of_bits,
                current_seed
            )
        )

        generated += current_batch_size
        batch_number += 1

    return jobs


def benchmark_parallel_simulator(
    number_of_sessions=100_000,
    number_of_bits=1000,
    batch_size=5000,
    workers=4,
    starting_seed=42
):
    """
    Benchmark exact BB84 generation using multiple processes.
    """
    if number_of_sessions <= 0:
        raise ValueError(
            "number_of_sessions must be greater than zero."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if workers <= 0:
        raise ValueError(
            "workers must be greater than zero."
        )

    available_cpus = os.cpu_count() or 1

    workers = min(
        workers,
        available_cpus
    )

    jobs = create_batch_jobs(
        number_of_sessions=number_of_sessions,
        batch_size=batch_size,
        number_of_bits=number_of_bits,
        starting_seed=starting_seed
    )

    print("=" * 65)
    print("PARALLEL EXACT BB84 BENCHMARK")
    print("=" * 65)

    print(
        f"Logical CPU cores  : "
        f"{available_cpus}"
    )

    print(
        f"Workers used       : "
        f"{workers}"
    )

    print(
        f"Sessions requested : "
        f"{number_of_sessions:,}"
    )

    print(
        f"Photons per session: "
        f"{number_of_bits:,}"
    )

    print(
        f"Batch size         : "
        f"{batch_size:,}"
    )

    print(
        f"Number of batches  : "
        f"{len(jobs):,}"
    )

    start_time = time.perf_counter()

    completed_sessions = 0
    dataframes = []

    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:

        results = executor.map(
            generate_worker,
            jobs
        )

        for batch_number, dataframe in enumerate(
            results,
            start=1
        ):
            dataframes.append(dataframe)

            completed_sessions += len(dataframe)

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            current_speed = (
                completed_sessions
                / elapsed_time
            )

            print(
                f"Completed batch "
                f"{batch_number:,}/{len(jobs):,} | "
                f"{completed_sessions:,}/"
                f"{number_of_sessions:,} sessions | "
                f"{current_speed:,.0f} sessions/sec"
            )

    final_dataframe = pd.concat(
        dataframes,
        ignore_index=True
    )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    final_speed = (
        len(final_dataframe)
        / elapsed_time
    )

    memory_mb = (
        final_dataframe.memory_usage(
            deep=True
        ).sum()
        / 1024**2
    )

    print("\n" + "=" * 65)
    print("PARALLEL BENCHMARK COMPLETE")
    print("=" * 65)

    print(
        f"Sessions generated : "
        f"{len(final_dataframe):,}"
    )

    print(
        f"Generation time    : "
        f"{elapsed_time:.4f} seconds"
    )

    print(
        f"Generation speed   : "
        f"{final_speed:,.0f} sessions/sec"
    )

    print(
        f"DataFrame memory   : "
        f"{memory_mb:.2f} MB"
    )

    print("\nLabel counts:")

    print(
        final_dataframe[
            "label"
        ].value_counts()
    )

    print("\nAverage feature values:")

    print(
        final_dataframe[
            [
                "qber",
                "photon_loss_rate",
                "detection_rate",
                "plus_basis_qber",
                "cross_basis_qber",
                "qber_variation"
            ]
        ].mean()
    )

    print("\nFirst five rows:")

    print(
        final_dataframe.head()
    )

    print("=" * 65)

    return final_dataframe


if __name__ == "__main__":
    benchmark_parallel_simulator(
        number_of_sessions=200_000,
        number_of_bits=1000,
        batch_size=5000,
        workers=6,
        starting_seed=42
    )