from decimal import Decimal, ROUND_HALF_UP
from functools import reduce
from pathlib import Path

import pandas as pd


def filter_high_risk(records):

    return filter(
        lambda record: (
            record["Overstay_Mins"] > 20
            and record["Battery_Temp_C"] > 40
        ),
        records
    )


def calculate_surcharge(record):
    """Return a new billing record with a 15 percent surcharge."""

    # Convert through a string to avoid importing binary float artifacts.
    base_fee = Decimal(str(record["Base_Fee_MYR"]))

    # Round each session's surcharge to the nearest cent using half-up rounding.
    surcharge = (base_fee * Decimal("0.15")).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP
    )

    # Base fee plus a 15% surcharge is equivalent to a 1.15 multiplier
    # before currency rounding. Existing peak and overstay fees are not added.
    adjusted_fee = base_fee + surcharge

    # Return a new dictionary without modifying the original record.
    # Keep only the fields needed for this question.
    return {
        "Session_ID": record["Session_ID"],
        "Base_Fee_MYR": base_fee,
        "Surcharge_MYR": surcharge,
        "Adjusted_Fee_MYR": adjusted_fee
    }


def total_surcharge(processed_records):

    return reduce(
        lambda total, record: total + record["Surcharge_MYR"],
        processed_records,
        Decimal("0.00")
    )


def main():
    """Read charging records, process high-risk sessions and print results."""

    # Read the CSV from the parent of the folder containing this script.
    parent_folder = Path(__file__).resolve().parent.parent
    file_path = parent_folder / "cleaned_energy_sales_data.csv"

    # Read only the columns required for this question.
    required_columns = [
        "Session_ID",
        "Overstay_Mins",
        "Battery_Temp_C",
        "Base_Fee_MYR"
    ]

    try:
        data = pd.read_csv(file_path, usecols=required_columns)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Unable to load the dataset: {error}")
        return

    # The supplied cleaned data should contain no missing required values.
    if data[required_columns].isna().any().any():
        print("Required values are missing. Clean the data before processing.")
        return

    # Convert the DataFrame into the list of session records required by A2.
    # This input list uses memory proportional to the number of records.
    records = data.to_dict(orient="records")

    # Step 1: Filter sessions using both risk conditions.
    high_risk_sessions = filter_high_risk(records)

    # Step 2: Apply the calculation to each matching session.
    # map() is lazy and consumes the filter iterator one record at a time.
    mapped_sessions = map(calculate_surcharge, high_risk_sessions)

    processed_sessions = list(mapped_sessions)

    # Step 3: Sum the per-session surcharges.
    surcharge_revenue = total_surcharge(processed_sessions)

    print("=== High-Risk Charging Sessions ===")
    print(f"Total input sessions: {len(records)}")
    print(f"High-risk sessions: {len(processed_sessions)}")

    print("\nFirst five processed sessions:")
    for record in processed_sessions[:5]:
        print(
            f"{record['Session_ID']} | "
            f"Base: RM {record['Base_Fee_MYR']:.2f} | "
            f"Surcharge: RM {record['Surcharge_MYR']:.2f} | "
            f"Adjusted: RM {record['Adjusted_Fee_MYR']:.2f}"
        )

    print(f"\nTotal surcharge revenue: RM {surcharge_revenue:.2f}")


if __name__ == "__main__":
    main()