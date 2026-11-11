import numpy as np
import pandas as pd

def main():
# Load the original sessions for temperature filtering and missing-value handling.
    raw_data = pd.read_csv("../ev_campus_dataset.csv")

    # Load the supplied cleaned dataset for existing billing amounts.
    cleaned_data = pd.read_csv("../cleaned_energy_sales_data.csv")

    # EXAMPLE DATA ONLY:
    # The supplied CSV files do not contain measured hourly campus grid demand.
    # These values are illustrative and are not estimated from charging sessions.
    # Rows represent two example days; columns represent hours 00 through 23.
    # All values are in kW.
    example_hourly_grid_kw = np.array([
        [100, 95, 90, 88, 90, 100, 120, 150,
         180, 200, 220, 240, 260, 280, 300, 290,
         270, 250, 220, 190, 160, 140, 120, 110],

        [105, 98, 92, 90, 95, 110, 130, 160,
         190, 210, 230, 250, 270, 290, 310, 300,
         280, 260, 230, 200, 170, 150, 130, 115]
    ], dtype=float)

    # Use 12:00-17:59 as an illustrative peak window for this example only.
    # This does not establish the actual campus peak-hour policy.
    # Select all rows and columns 12 through 17; the stop index 18 is excluded.
    example_peak_load = example_hourly_grid_kw[:, 12:18]

    print("EXAMPLE peak-hour grid demand (kW):")
    print(example_peak_load)

    # Convert the supplied battery temperature column into a NumPy array.
    battery_temperatures = raw_data["Battery_Temp_C"].to_numpy(dtype=float)

    # Create a Boolean mask; readings equal to 45 degrees Celsius are excluded.
    hot_mask = battery_temperatures > 45

    # Select only the temperatures that satisfy the condition.
    hot_temperatures = battery_temperatures[hot_mask]

    print("Number of readings above 45 degrees Celsius:", hot_temperatures.size)

    # Display only the first ten matches to keep the output readable.
    print("First ten matching temperatures:")
    print(hot_temperatures[:10])

    # Work on a copy so the original DataFrame remains available for comparison.
    working_data = raw_data.copy()

    # Count missing energy values before imputation.
    missing_before = working_data["Energy_kWh"].isna().sum()

    # Calculate the median from the available energy values.
    median_energy = working_data["Energy_kWh"].median()

    # A completely missing column would not provide a valid median.
    if pd.isna(median_energy):
        raise ValueError("Energy_kWh has no valid values for median imputation.")

    # Replace missing energy values with the column median.
    # Filled values are statistical substitutes, not measured observations.
    working_data["Energy_kWh"] = (
        working_data["Energy_kWh"].fillna(median_energy)
    )

    missing_after = working_data["Energy_kWh"].isna().sum()

    print("Missing energy values before:", missing_before)
    print("Median energy (kWh):", median_energy)
    print("Missing energy values after:", missing_after)

    # Use the supplied net billable amounts without recalculating the tariff.
    # These amounts represent billed charges, not confirmed payment receipts.
    # Existing amounts may incorporate previously imputed session values.
    billable_by_type = (
        cleaned_data
        .groupby("Charger_Type")["Net_Billable_MYR"]
        .sum()
    )

    print("Total net billable amount by charger type (MYR):")

    # Format the displayed amounts to two decimal places.
    print(billable_by_type.to_string(float_format=lambda value: f"{value:.2f}"))

    print(f"\nOverall net billable amount (MYR): {billable_by_type.sum():.2f}")

if __name__ == "__main__":
    main()