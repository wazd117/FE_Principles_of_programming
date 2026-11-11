from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_hourly_revenue(data, output_folder):
    """Create a grouped bar chart of billable amounts by start hour."""

    # Use the existing billable amounts without recalculating fees.
    # Attribute each session's full bill to its start-hour category.
    # This is not a measurement of revenue earned within each clock hour.
    hourly_revenue = (
        data.groupby(["Start_Hour", "Charger_Type"])["Net_Billable_MYR"]
        .sum()
        .unstack(fill_value=0)
    )

    # Keep all 24 hours and both charger types in a consistent order.
    hourly_revenue = hourly_revenue.reindex(
        index=range(24),
        columns=["AC", "DC"],
        fill_value=0
    )

    hours = np.arange(24)
    bar_width = 0.38

    fig, ax = plt.subplots(
        figsize=(13, 6),
        layout="constrained"
    )

    # Offset the bars so AC and DC appear beside each other.
    ax.bar(
        hours - bar_width / 2,
        hourly_revenue["AC"],
        width=bar_width,
        label="AC",
        color="#0072B2"
    )

    ax.bar(
        hours + bar_width / 2,
        hourly_revenue["DC"],
        width=bar_width,
        label="DC",
        color="#D55E00"
    )

    ax.set_title(
        "Charging revenue by session start hour and charger type"
    )
    ax.set_xlabel("Session start hour (00-23)")
    ax.set_ylabel("Total net billable amount (MYR)")
    ax.set_xticks(hours)

    # Start the vertical axis at zero for fair bar-height comparisons.
    ax.set_ylim(bottom=0)
    ax.legend(title="Charger type")

    # Light horizontal grid lines help readers compare amounts.
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    output_path = output_folder / "hourly_charging_revenue.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print("\nTotal net billable amount by charger type (MYR):")
    print(hourly_revenue.sum().round(2).to_string())
    print(f"Bar chart saved to: {output_path}")

    return fig


def plot_correlation_heatmap(data, output_folder):
    """Create an illustrative correlation heatmap with labelled example data."""

    # EXAMPLE DATA ONLY:
    # Ambient temperature and numerical grid load are absent from the files.
    # Generate illustrative values; these are not estimates or measurements.
    # The ranges below are demonstration choices, not campus specifications.
    # A fixed seed makes the example reproducible.
    rng = np.random.default_rng(72604)

    example_ambient_temp = rng.uniform(24, 36, size=len(data))
    example_grid_load = rng.uniform(100, 300, size=len(data))

    # Preserve charging durations from the supplied cleaned dataset.
    # Pairing example values with these sessions is for plotting practice only.
    # Correlations involving the example columns are not campus findings.
    heatmap_data = pd.DataFrame({
        "Ambient temperature (C)": example_ambient_temp,
        "Charging duration (min)": data["Duration_Mins"].to_numpy(),
        "Grid load (kW)": example_grid_load
    })

    correlation_matrix = heatmap_data.corr(method="pearson")

    fig, ax = plt.subplots(
        figsize=(9, 7),
        layout="constrained"
    )

    # Use a diverging colormap centered at zero.
    # Fix the scale from -1 to 1 so colors have a consistent meaning.
    # Display coefficients to two decimal places inside the cells.
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation (r)"},
        ax=ax
    )

    # Disclose the example data in the figure itself.
    ax.set_title(
        "Illustrative correlation heatmap\n"
        "EXAMPLE ambient temperature and grid load; supplied duration",
        pad=16
    )

    ax.tick_params(axis="x", rotation=15)
    ax.tick_params(axis="y", rotation=0)

    output_path = output_folder / "illustrative_correlation_heatmap.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print("\nIllustrative correlation matrix:")
    print(correlation_matrix.round(3).to_string())
    print("Example-data correlations must not be treated as campus findings.")
    print(f"Heatmap saved to: {output_path}")

    return fig


def main():
    """Load the supplied data, create both charts and display them."""

    script_folder = Path(__file__).resolve().parent
    input_path = script_folder.parent / "cleaned_energy_sales_data.csv"
    output_folder = script_folder / "outputs"

    required_columns = [
        "Start_Hour",
        "Charger_Type",
        "Net_Billable_MYR",
        "Duration_Mins"
    ]

    try:
        data = pd.read_csv(input_path, usecols=required_columns)
        output_folder.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Unable to prepare the charts: {error}")
        return

    if data.empty or data[required_columns].isna().any().any():
        print("The required data is empty or contains missing values.")
        return

    revenue_figure = plot_hourly_revenue(data, output_folder)
    correlation_figure = plot_correlation_heatmap(data, output_folder)

    # Display both figures when running in a graphical Python environment.
    plt.show()

    plt.close(revenue_figure)
    plt.close(correlation_figure)


if __name__ == "__main__":
    main()