"""CartPoleCost Environment Translation Validation Script.

This script validates the translation of the 'CartPoleCost' environment from the
'Actor-critic-with-stability-guarantee' repository to the 'stable_gym' package. It does
this by comparing the output of the 'step' method in the translated environment with the
output from the original implementation.

The original output should be placed in a CSV file located in the same directory as this
script. This CSV file is generated by a corresponding script in the
'Actor-critic-with-stability-guarantee' repository.

For detailed usage instructions, please refer to the README.md in this directory.
"""
import stable_gym  # NOTE: Ensures that the latest version of the environment is used. # noqa: F401, E501
import numpy as np
from prettytable import PrettyTable
import textwrap
import gymnasium as gym
import os
import pandas as pd
import math

STEPS = 10
CHECK_TOLERANCE = 1e-7
SEED = 0
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_accuracy(number):
    """Get the number of decimal places of a number.

    Args:
        number (float): The number to get the decimal places of.

    Returns:
        int: The number of decimal places of the number.
    """
    number_str = str(number)
    parts = number_str.split(".")

    # If there's no decimal point, the number is an integer.
    if len(parts) == 1:
        return 0  # An integer has zero decimal places.

    # Count the number of digits after the decimal point and return.
    decimal_places = len(parts[1])
    return decimal_places


if __name__ == "__main__":
    print("===CartPoleCost Environment Translation Validation===")
    print(
        textwrap.dedent(
            f"""
            Welcome to the CartPoleCost environment translation validation script. This
            script initializes the stable-gym CartPoleCost environment and executes
            '{STEPS}' steps to validate in against the original implementation in the
            'Actor-critic-with-stability-guarantee' repository.

            Compare the output of this script with the equivalent script in the
            'Actor-critic-with-stability-guarantee' repository. Matching outputs
            indicate  correct translations.

            Note: These script only compares step outputs, not reset outputs. This is
            due to non-deterministic behaviors between different numpy and gym versions
            used in both packages.
            """
        )
    )

    # Initialize CartPoleCost environment.
    # NOTE: The state is set directly due to non-deterministic behaviour across different
    # numpy and gym versions.
    env_cost = gym.make(
        "CartPoleCost",
        action_space_dtype=np.float32,
        observation_space_dtype=np.float32,
    )
    env_cost = env_cost.unwrapped
    env_cost.reset(seed=SEED)
    env_cost.unwrapped.state = np.array(
        [
            0.35412586,
            -45.697277,
            -0.15257455,
            32.553246,
        ],
        dtype=np.float32,
    )

    # Create a pretty table to display the results in.
    table = PrettyTable()
    table.field_names = [
        "Step",
        "Obs1",
        "Obs2",
        "Obs3",
        "Obs4",
        "Reward",
        "Done",
        "Reference",
        "State of Interest",
    ]

    # Perform N steps for the stable-gym environment comparison.
    # NOTE: Use the same action as in the stable-gym package.
    df = pd.DataFrame(
        columns=[
            "Step",
            "Obs1",
            "Obs2",
            "Obs3",
            "Obs4",
            "Reward",
            "Done",
            "Reference",
            "StateOfInterest",
        ]
    )
    for i in range(STEPS):
        delta = (
            (env_cost.action_space.high - env_cost.action_space.low)[0] / STEPS
        ) * i
        action = np.array([env_cost.action_space.low[0] + delta], dtype=np.float32)
        observation, reward, terminated, truncated, info = env_cost.step(action)

        # Store the results in a table.
        done = terminated or truncated
        reference = info["reference"]
        state_of_interest = info["state_of_interest"]
        table.add_row(
            [
                i,
                observation[0],
                observation[1],
                observation[2],
                observation[3],
                reward,
                done,
                reference,
                state_of_interest,
            ]
        )
        new_row = pd.DataFrame(
            {
                "Step": [np.int64(i)],
                "Obs1": [observation[0]],
                "Obs2": [observation[1]],
                "Obs3": [observation[2]],
                "Obs4": [observation[3]],
                "Reward": [reward],
                "Done": [done],
                "Reference": [reference],
                "StateOfInterest": [state_of_interest],
            }
        )
        df = pd.concat([df, new_row], ignore_index=True)

    # Save the results to a CSV file.
    df.to_csv(
        os.path.join(
            SCRIPT_DIR, "results/stableGym_cartPoleCost_translation_validation.csv"
        )
    )
    env_cost.close()
    print(
        "Stable gym cartPoleCost comparison table generated and stored in "
        "'results/stableGym_cartPoleCost_translation_validation.csv'."
    )

    # Print the results of the stable-gym environment steps.
    print("\nStable gym CartPoleCost comparison table:")
    print(f"{table}\n")

    # Check if the reference CSV file exists.
    if not os.path.isfile(
        os.path.join(SCRIPT_DIR, "results/cartPoleCost_translation_validation.csv")
    ):
        print(
            "\nNo 'results/cartPoleCost_translation_validation.csv' file found. Please "
            "run the same script in the 'Actor-critic-with-stability-guarantee' "
            "repository to generate the file and place it in the 'results' folder "
            "found alongside this script to get a comparison result."
        )
        exit()

    # Load the reference fil CSV file.
    df2 = pd.read_csv(
        os.path.join(SCRIPT_DIR, "results/cartPoleCost_translation_validation.csv"),
    )

    # Print the reference CSV file results as a pretty table.
    table2 = PrettyTable()
    table2.field_names = [
        "Step",
        "Obs1",
        "Obs2",
        "Obs3",
        "Obs4",
        "Reward",
        "Done",
        "Reference",
        "StateOfInterest",
    ]
    df2_tmp = df2.round(7)
    for i, row in df2_tmp.iterrows():
        table2.add_row(row)
    print("\nReference CSV file table:")
    print(f"{table2}\n")

    # Ensure results are compatible.
    # NOTE: Some data was not returned by the original implementation or used a different
    # data type.
    df["Reference"] = df["Reference"].apply(lambda x: x[0])
    df["StateOfInterest"] = df["StateOfInterest"].apply(lambda x: x[-1])
    df2["Reference"] = df2["Reference"].apply(lambda x: float(x))

    # Compare the results.
    print(
        "\nComparing stable-gym step results with results in the "
        "'results/cartPoleCost_translation_validation.csv' file."
    )
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    boolean_cols = df.select_dtypes(include=["bool"]).columns
    numeric_close = np.allclose(
        df[numeric_cols].values,
        df2[numeric_cols].values,
        atol=CHECK_TOLERANCE,
        equal_nan=True,
    )
    boolean_equal = (df[boolean_cols] == df2[boolean_cols]).all().all()
    accuracy = min(get_accuracy(df2["Obs1"][0]), get_accuracy(df["Obs1"][0]))
    if numeric_close and boolean_equal:
        print(
            "✅ Test Passed: Results are consistent up to a precision of "
            f"{min(accuracy, abs(math.log10(CHECK_TOLERANCE)))} decimal places."
        )
    else:
        raise ValueError("❌ Test Failed: Results do not match the expected values.")