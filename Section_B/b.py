from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_energy_curves(ax):
    """Plot illustrative hourly grid demand and solar generation."""

    # EXAMPLE DATA ONLY:
    # The supplied datasets contain no measured hourly grid demand
    # or solar generation. These values are illustrative, not estimates.
    # The horizontal axis represents hours in an example day.
    hours = np.arange(24)

    example_grid_kw = np.array([
        100, 95, 90, 88, 90, 100, 120, 150,
        180, 200, 220, 240, 260, 280, 300, 290,
        270, 250, 220, 190, 160, 140, 120, 110
    ], dtype=float)

    example_solar_kw = np.array([
        0, 0, 0, 0, 0, 0, 5, 15,
        35, 55, 75, 90, 100, 95, 80, 60,
        35, 15, 3, 0, 0, 0, 0, 0
    ], dtype=float)

    # Use different colors, markers and line styles for clear distinction.
    ax.plot(
        hours,
        example_grid_kw,
        label="Grid demand (example)",
        color="#0072B2",
        marker="o",
        markersize=4
    )

    ax.plot(
        hours,
        example_solar_kw,
        label="Solar generation (example)",
        color="#D55E00",
        linestyle="--",
        marker="s",
        markersize=4
    )

    # Identify and annotate the maximum value in the example grid series.
    peak_index = int(np.argmax(example_grid_kw))
    peak_hour = hours[peak_index]
    peak_value = example_grid_kw[peak_index]

    ax.annotate(
        f"Example peak: {peak_value:.0f} kW",
        xy=(peak_hour, peak_value),
        xytext=(peak_hour + 2, peak_value + 35),
        arrowprops={
            "arrowstyle": "->",
            "color": "black"
        }
    )

    ax.set_title(
        "Panel 1: Hourly grid demand and solar generation\n"
        "EXAMPLE DATA ONLY - illustrative day, not campus measurements"
    )
    ax.set_xlabel("Hour of example day")
    ax.set_ylabel("Power (kW)")
    ax.set_xticks(hours)
    ax.set_ylim(0, 370)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


def plot_session_costs(data, ax):
    """Compare session cost distributions across user types."""

    # Use existing net billable amounts from the cleaned dataset.
    # Each observation represents one charging session.
    sns.boxplot(
        data=data,
        x="User_Type",
        y="Net_Billable_MYR",
        order=["Student", "Staff", "Visitor"],
        color="#9AC9DF",
        ax=ax
    )

    # Keep outlier markers visible; unusual costs are not automatically errors.
    ax.set_title("Panel 2: Charging session costs by user type")
    ax.set_xlabel("User type")
    ax.set_ylabel("Net billable amount (MYR)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def plot_temperature_energy(data, ax):
    """Plot battery temperature against energy with a linear trend."""

    # Use the supplied temperature and cleaned energy values.
    # Some energy values were median-imputed during data cleaning.
    # regplot() draws both the observations and an ordinary least-squares line.
    # The overall trend describes association, not causation.
    sns.regplot(
        data=data,
        x="Battery_Temp_C",
        y="Energy_kWh",
        ci=None,
        scatter_kws={
            "alpha": 0.4,
            "s": 20,
            "color": "#0072B2"
        },
        line_kws={
            "color": "#D55E00",
            "linewidth": 2,
            "label": "Overall linear trend"
        },
        ax=ax
    )

    ax.set_title("Panel 3: Battery temperature and energy delivered")
    ax.set_xlabel("Battery temperature (°C)")
    ax.set_ylabel("Energy delivered (kWh)")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


def main():
    """Load cleaned sessions and create the three required graphical panels."""

    script_folder = Path(__file__).resolve().parent
    output_folder = script_folder / "outputs"

    # Read the dataset exported by Section B(a).
    input_path = output_folder / "cleaned_energy_sales_data.csv"

    required_columns = [
        "User_Type",
        "Net_Billable_MYR",
        "Battery_Temp_C",
        "Energy_kWh"
    ]

    try:
        data = pd.read_csv(input_path, usecols=required_columns)

        if data.empty:
            raise ValueError("The cleaned dataset is empty.")

        if data[required_columns].isna().any().any():
            raise ValueError("Required columns contain missing values.")

        allowed_users = ["Student", "Staff", "Visitor"]
        if not data["User_Type"].isin(allowed_users).all():
            raise ValueError("Unknown user types were found.")

        numeric_columns = [
            "Net_Billable_MYR",
            "Battery_Temp_C",
            "Energy_kWh"
        ]

        for column in numeric_columns:
            data[column] = pd.to_numeric(data[column], errors="raise")

        if not np.isfinite(data[numeric_columns].to_numpy()).all():
            raise ValueError("Numeric columns contain non-finite values.")

        if len(data) < 2 or data["Battery_Temp_C"].nunique() < 2:
            raise ValueError(
                "Insufficient temperature variation for a linear trend."
            )

        # Create three vertically arranged panels with automatic spacing.
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(12, 15),
            layout="constrained"
        )

        plot_energy_curves(axes[0])
        plot_session_costs(data, axes[1])
        plot_temperature_energy(data, axes[2])

        output_folder.mkdir(parents=True, exist_ok=True)

        # Save a high-resolution image and a vector PDF for report use.
        png_path = output_folder / "section_b_three_panels.png"
        pdf_path = output_folder / "section_b_three_panels.pdf"

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")

        print("Three graphical panels created successfully.")
        print(f"Charging sessions used in Panels 2 and 3: {len(data)}")
        print("Panel 1 uses illustrative example data only.")
        print(f"PNG saved to: {png_path}")
        print(f"PDF saved to: {pdf_path}")

        plt.show()
        plt.close(fig)

    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Visualization failed: {error}")


if __name__ == "__main__":
    main()