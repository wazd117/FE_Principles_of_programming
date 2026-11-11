from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "Start_Hour",
    "Duration_Mins",
    "Energy_kWh",
    "Overstay_Mins",
    "Battery_Temp_C"
]

ALLOWED_CATEGORIES = {
    "Charger_Type": ["AC", "DC"],
    "User_Type": ["Student", "Staff", "Visitor"],
    "Grid_Status_Label": ["Normal", "Grid_Surge", "Overheated"]
}


def load_data(file_path):
    """Read the original CSV and check its required fields."""

    data = pd.read_csv(file_path, dtype={"Session_ID": "string"})

    required_columns = (
        ["Session_ID"]
        + list(ALLOWED_CATEGORIES)
        + NUMERIC_COLUMNS
    )

    missing_columns = set(required_columns) - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("The dataset contains no records.")

    # Keep the original fields and rebuild derived metrics later.
    data = data[required_columns].copy()

    data["Session_ID"] = data["Session_ID"].str.strip()

    if data["Session_ID"].isna().any() or data["Session_ID"].eq("").any():
        raise ValueError("Missing session identifiers were found.")

    # Duplicate identifiers require investigation before financial aggregation.
    if data["Session_ID"].duplicated().any():
        raise ValueError("Duplicate Session_ID values were found.")

    return data


def clean_data(data):
    """Validate categories and impute missing numerical values."""

    cleaned = data.copy()

    # Normalize capitalization and surrounding spaces.
    # Reject unknown categories instead of assigning an arbitrary category.
    for column, allowed_values in ALLOWED_CATEGORIES.items():
        mapping = {
            value.casefold(): value
            for value in allowed_values
        }

        normalized = (
            cleaned[column]
            .astype("string")
            .str.strip()
            .str.casefold()
        )

        invalid_mask = ~normalized.isin(mapping)

        if invalid_mask.any():
            invalid_values = (
                cleaned.loc[invalid_mask, column]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                f"Invalid values in {column}: {invalid_values}"
            )

        cleaned[column] = normalized.map(mapping)

    print("\n=== Numerical Data Cleaning ===")

    for column in NUMERIC_COLUMNS:
        # Convert invalid numeric text and infinite values into missing values.
        values = pd.to_numeric(cleaned[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)

        # A positive duration is required to calculate power safely.
        if column == "Duration_Mins":
            values = values.where(values > 0)

        # Negative energy and overstay are invalid for this dataset.
        elif column in ["Energy_kWh", "Overstay_Mins"]:
            values = values.where(values >= 0)

        # Start_Hour represents a whole hour from 0 through 23.
        elif column == "Start_Hour":
            values = values.where(
                values.between(0, 23) & values.mod(1).eq(0)
            )

        missing_count = int(values.isna().sum())
        median_value = values.median()

        if pd.isna(median_value):
            raise ValueError(
                f"No valid median is available for {column}."
            )

        # Impute missing numerical values using the column median.
        cleaned[column] = values.fillna(median_value)

        print(
            f"{column}: filled {missing_count} values; "
            f"median = {median_value:.2f}"
        )

    # Do not silently round a fractional median used to fill Start_Hour.
    if not cleaned["Start_Hour"].mod(1).eq(0).all():
        raise ValueError(
            "Median imputation produced a non-integer Start_Hour."
        )

    cleaned["Start_Hour"] = cleaned["Start_Hour"].astype(int)

    return cleaned


def calculate_metrics(data):
    """Calculate energy metrics and charges using the supplied reference rules."""

    result = data.copy()

    # Convert minutes to hours before calculating average session power.
    result["Duration_Hours"] = result["Duration_Mins"] / 60

    result["Power_Rate_kW"] = (
        result["Energy_kWh"] / result["Duration_Hours"]
    )

    # Use the carbon factor specified in the assessment.
    result["Carbon_Savings_kgCO2"] = (
        result["Energy_kWh"] * 0.475
    ).round(2)

    # These tariff rules were verified against the supplied cleaned CSV.
    # They are not independently verified official campus policies.
    # Base_Rate_MYR is an hourly rate, not a price per kWh.
    hourly_rates = {"AC": 6.0, "DC": 15.0}

    result["Base_Rate_MYR"] = result["Charger_Type"].map(hourly_rates)

    result["Base_Fee_MYR"] = (
        result["Duration_Hours"] * result["Base_Rate_MYR"]
    )

    # The reference file applies a 25% surcharge to sessions starting
    # at hours 12 through 17, inclusive.
    peak_mask = result["Start_Hour"].between(12, 17)

    result["Peak_Surge_MYR"] = np.where(
        peak_mask,
        result["Base_Fee_MYR"] * 0.25,
        0.0
    )

    # The reference file allows 15 overstay minutes without a fee.
    # Each additional minute costs MYR 0.50.
    billable_overstay = (
        result["Overstay_Mins"] - 15
    ).clip(lower=0)

    result["Overstay_Fee_MYR"] = billable_overstay * 0.50

    # Preserve intermediate precision and round the final bill.
    # This calculation order reproduces the supplied reference amounts.
    result["Net_Billable_MYR"] = (
        result["Base_Fee_MYR"]
        + result["Peak_Surge_MYR"]
        + result["Overstay_Fee_MYR"]
    ).round(2)

    return result


def main():
    """Run the pipeline, export the cleaned data and print summary statistics."""

    script_folder = Path(__file__).resolve().parent
    input_path = script_folder.parent / "ev_campus_dataset.csv"

    # Save generated results separately from the supplied reference CSV.
    output_folder = script_folder / "outputs"
    output_path = output_folder / "cleaned_energy_sales_data.csv"

    try:
        original_data = load_data(input_path)
        cleaned_data = clean_data(original_data)
        final_data = calculate_metrics(cleaned_data)

        output_folder.mkdir(parents=True, exist_ok=True)
        final_data.to_csv(output_path, index=False)

        summary_columns = [
            "Duration_Mins",
            "Energy_kWh",
            "Battery_Temp_C",
            "Power_Rate_kW",
            "Carbon_Savings_kgCO2",
            "Net_Billable_MYR"
        ]

        print("\n=== Summary Statistics ===")
        print(
            final_data[summary_columns]
            .describe()
            .round(2)
            .to_string()
        )

        print("\n=== Category Validation Results ===")
        for column in ALLOWED_CATEGORIES:
            print(f"\n{column}:")
            print(final_data[column].value_counts().to_string())

        print("\n=== Processing Results ===")
        print(f"Input records: {len(original_data)}")
        print(f"Output records: {len(final_data)}")
        print(
            "Remaining missing values:",
            int(final_data.isna().sum().sum())
        )
        print(
            "Total net billable amount (MYR): "
            f"{final_data['Net_Billable_MYR'].sum():.2f}"
        )
        print(f"Cleaned dataset saved to: {output_path}")

    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Data processing failed: {error}")


if __name__ == "__main__":
    main()