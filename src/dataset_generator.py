import os
import random
import time

import pandas as pd

from bb84_simulator import run_bb84


def generate_dataset(
    number_of_sessions=100_000,
    output_file="datasets/bb84_dataset.csv"
):
    """
    Generate a BB84 machine-learning dataset.

    Half of the sessions are approximately normal communication,
    and half contain an Eve intercept-resend attack.
    """
    dataset = []

    print(
        f"Generating {number_of_sessions:,} "
        f"exact BB84 sessions...\n"
    )

    output_directory = os.path.dirname(output_file)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    start_time = time.perf_counter()

    for i in range(number_of_sessions):
        eve_present = random.choice([True, False])

        if eve_present:
            interception_probability = random.uniform(
                0.10,
                1.00
            )
        else:
            interception_probability = 0.0

        noise_probability = random.uniform(
            0.00,
            0.05
        )

        photon_loss_probability = random.uniform(
            0.00,
            0.10
        )

        features = run_bb84(
            number_of_bits=1000,
            eve_present=eve_present,
            interception_probability=interception_probability,
            noise_probability=noise_probability,
            photon_loss_probability=photon_loss_probability,
            show_output=False
        )

        features["interception_probability"] = (
            interception_probability
        )

        features["label"] = 1 if eve_present else 0

        dataset.append(features)

        if (i + 1) % 5000 == 0:
            elapsed_time = (
                time.perf_counter() - start_time
            )

            speed = (i + 1) / elapsed_time

            print(
                f"Generated {i + 1:,}/"
                f"{number_of_sessions:,} sessions "
                f"({speed:,.0f} sessions/sec)"
            )

    dataframe = pd.DataFrame(dataset)

    dataframe = dataframe.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    dataframe.to_csv(
        output_file,
        index=False
    )

    total_time = time.perf_counter() - start_time

    average_speed = (
        number_of_sessions / total_time
    )

    print("\n====================================")
    print("Dataset Successfully Created!")
    print("====================================")
    print(f"Saved to       : {output_file}")
    print(f"Total sessions : {len(dataframe):,}")
    print(f"Total time     : {total_time:.2f} seconds")
    print(
        f"Average speed  : "
        f"{average_speed:,.0f} sessions/sec"
    )

    print("\nDataset columns:")
    print(dataframe.columns.tolist())

    print("\nFirst five rows:")
    print(dataframe.head())


if __name__ == "__main__":
    generate_dataset(
        number_of_sessions=100_000,
        output_file="datasets/bb84_dataset_numpy.csv"
    )