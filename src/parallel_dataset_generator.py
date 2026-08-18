"""
Parallel BB84 Dataset Generator

Generates an exact BB84 dataset using multiple CPU cores.
Each completed batch is immediately written to disk so
memory usage remains nearly constant.

Author: Mani
"""

import gc
import os
import time

from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from bb84_batch_simulator import generate_exact_bb84_batch


# ==========================================================
# Configuration
# ==========================================================

NUMBER_OF_SESSIONS = 10_000_000

NUMBER_OF_BITS = 1000

BATCH_SIZE = 5000

WORKERS = 6

OUTPUT_FILE = "datasets/bb84_dataset_10M.csv"

STARTING_SEED = 42


# ==========================================================
# Worker Function
# ==========================================================

def generate_worker(arguments):
    """
    Generate one batch of exact BB84 sessions.
    """

    batch_size, number_of_bits, seed = arguments

    dataframe = generate_exact_bb84_batch(
        number_of_sessions=batch_size,
        number_of_bits=number_of_bits,
        seed=seed
    )

    return dataframe


# ==========================================================
# Job Creator
# ==========================================================

def create_jobs():
    """
    Divide the requested dataset into many independent jobs.
    """

    jobs = []

    generated = 0

    batch_number = 0

    while generated < NUMBER_OF_SESSIONS:

        current_batch = min(
            BATCH_SIZE,
            NUMBER_OF_SESSIONS - generated
        )

        jobs.append(
            (
                current_batch,
                NUMBER_OF_BITS,
                STARTING_SEED + batch_number
            )
        )

        generated += current_batch

        batch_number += 1

    return jobs


# ==========================================================
# Progress Printer
# ==========================================================

def print_progress(
    completed,
    total,
    start_time
):
    """
    Print generation progress.
    """

    elapsed = time.perf_counter() - start_time

    speed = completed / elapsed

    remaining = total - completed

    eta = remaining / speed

    eta_minutes = int(eta // 60)

    eta_seconds = int(eta % 60)

    percent = completed / total * 100

    print(
        f"{completed:,}/{total:,} "
        f"({percent:.2f}%) | "
        f"{speed:,.0f} sessions/sec | "
        f"ETA {eta_minutes}m {eta_seconds}s"
    )
# ==========================================================
# Dataset Generator
# ==========================================================

def generate_dataset():
    """
    Generate the complete dataset and write directly to CSV.
    """

    jobs = create_jobs()

    total_sessions = NUMBER_OF_SESSIONS

    completed_sessions = 0

    start_time = time.perf_counter()

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    first_batch = True

    print("=" * 60)
    print("STARTING PARALLEL BB84 DATASET GENERATION")
    print("=" * 60)

    print(f"Output File : {OUTPUT_FILE}")
    print(f"Workers     : {WORKERS}")
    print(f"Batch Size  : {BATCH_SIZE:,}")
    print(f"Sessions    : {NUMBER_OF_SESSIONS:,}")
    print()

    with ProcessPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        results = executor.map(
            generate_worker,
            jobs
        )

        for dataframe in results:

            dataframe.to_csv(
                OUTPUT_FILE,
                mode="a",
                header=first_batch,
                index=False
            )

            first_batch = False

            completed_sessions += len(dataframe)

            print_progress(
                completed_sessions,
                total_sessions,
                start_time
            )

            del dataframe

            gc.collect()

    elapsed = time.perf_counter() - start_time

    print()
    print("=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"Total Sessions : "
        f"{completed_sessions:,}"
    )

    print(
        f"Total Time     : "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Average Speed  : "
        f"{completed_sessions / elapsed:,.0f} sessions/sec"
    )

    print(
        f"Dataset Saved  : "
        f"{OUTPUT_FILE}"
    )
# ==========================================================
# Main
# ==========================================================

def main():
    """
    Entry point.
    """

    print()
    print("=" * 60)
    print("BB84 PARALLEL DATASET GENERATOR")
    print("=" * 60)

    print(f"CPU Workers : {WORKERS}")
    print(f"Batch Size  : {BATCH_SIZE:,}")
    print(f"Output File : {OUTPUT_FILE}")

    estimated_batches = (
        NUMBER_OF_SESSIONS + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print(
        f"Total Batches : "
        f"{estimated_batches:,}"
    )

    print()

    generate_dataset()

    print()
    print("=" * 60)
    print("Generation Finished Successfully")
    print("=" * 60)


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("Generation cancelled by user.")

    except Exception as error:

        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print(error)

    finally:

        gc.collect()