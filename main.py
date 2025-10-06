# interactive_predict.py
"""
Command-line interactive script
---------------------------------
$ python interactive_predict.py
---------------------------------
"""
import os
import pandas as pd
from data_generation import generate_dataset
from train_model import train_model
from plot_candidate_score import plot_candidate_score_cool
import sys
from scorer import Scorer


def ask_float(field_name: str, mini: float = None, maxi: float = None) -> float | None:
    """
    Prompt the user for a numeric input and return a float.
    Returns None if the user enters q / quit / exit / empty line, signaling to exit the main loop.
    """
    tips = "(Press Enter or type q to quit)"
    if mini is not None and maxi is not None:
        tips += f" [{mini} ~ {maxi}]"
    prompt = f"Please enter {field_name} {tips}: "

    while True:
        raw = input(prompt).strip().lower()
        if raw in {"", "q", "quit", "exit"}:
            return None
        try:
            value = float(raw.replace(",", ""))  # Support numbers with commas
            if mini is not None and value < mini:
                print(f"❌ {field_name} cannot be less than {mini}")
                continue
            if maxi is not None and value > maxi:
                print(f"❌ {field_name} cannot be greater than {maxi}")
                continue
            return value
        except ValueError:
            print("❌ Invalid number, please try again.")


def main() -> None:
    dataset_path = "GaoFuShai_Dataset.xlsx"

    # Step 1: Check or generate dataset
    if not os.path.exists(dataset_path):
        df = generate_dataset(filename=dataset_path)
    else:
        print(f"Found existing dataset '{dataset_path}', loading it directly.")
        df = pd.read_excel(dataset_path)

    # Step 2: Train the model
    train_model(df)

    print("============== Real-time Handsome-Rich Score Prediction ==============")
    print("Please enter Height (cm), Wealth (CNY), and Looks (0-100) as prompted.")

    scorer = Scorer()

    while True:
        # 1️⃣ Read user input
        height = ask_float("Height (cm)", mini=50, maxi=300)
        if height is None:
            break

        wealth = ask_float("Wealth (CNY)", mini=0)
        if wealth is None:
            break

        looks = ask_float("Looks (0-100)", mini=0, maxi=100)
        if looks is None:
            break

        # 2️⃣ Calculate score
        score = scorer.calculate_score(height, wealth, looks)

        # 3️⃣ Output result
        print(f"\n>>> Your Handsome-Rich composite score is: {score:.2f} / 100\n")
        print("===================================================================")
        plot_candidate_score_cool(height, wealth, looks)
    print("\n👋 Exited. Handsome-Rich prediction ended.")
    sys.exit(0)


if __name__ == "__main__":
    main()
