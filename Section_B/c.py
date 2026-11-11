"""Section B(c): run with python -m streamlit run c.py."""

from datetime import timedelta
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


@st.cache_data(show_spinner=False)
def load_data(csv_bytes):
    """Validate the cleaned CSV and add explicitly labelled demo dates if needed."""
    data = pd.read_csv(BytesIO(csv_bytes))
    numeric = [
        "Energy_kWh",
        "Carbon_Savings_kgCO2",
        "Net_Billable_MYR",
        "Battery_Temp_C",
        "Overstay_Mins",
    ]
    required = ["Session_ID", "Charger_Type", "User_Type"] + numeric
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data.empty or data[required].isna().any().any():
        raise ValueError("Required data is empty or contains missing values.")
    if data["Session_ID"].duplicated().any():
        raise ValueError("Session_ID values must be unique.")
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data[numeric].to_numpy()).all():
        raise ValueError("Numeric values must be finite.")
    nonnegative = [column for column in numeric if column != "Battery_Temp_C"]
    if (data[nonnegative] < 0).any().any():
        raise ValueError(
            "Energy, fees, carbon savings and overstay cannot be negative."
        )
    if not data["Charger_Type"].isin(["AC", "DC"]).all():
        raise ValueError("Charger_Type must be AC or DC.")
    if not data["User_Type"].isin(["Student", "Staff", "Visitor"]).all():
        raise ValueError("Unknown User_Type values were found.")

    if "Session_Date" not in data.columns:
        # EXAMPLE DATES ONLY: the supplied charging CSV has no calendar dates.
        # Assign dates cyclically across an arbitrary 14-day demonstration.
        # Sorting IDs makes assignments stable if the input row order changes.
        # These are not inferred session dates and are unrelated to sales dates.
        data = data.sort_values("Session_ID").reset_index(drop=True)
        data["Session_Date"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(
            np.arange(len(data)) % 14, unit="D"
        )
        data["Date_Is_Example"] = True
    else:
        data["Session_Date"] = pd.to_datetime(
            data["Session_Date"], format="%Y-%m-%d", errors="raise"
        )
        if data["Session_Date"].isna().any():
            raise ValueError("Session_Date contains missing dates.")
        # Preserve the disclosure if a previously exported demo CSV is reused.
        if "Date_Is_Example" in data.columns:
            flag = data["Date_Is_Example"].astype(str).str.lower()
            if not flag.isin(["true", "false"]).all():
                raise ValueError("Date_Is_Example must contain True or False.")
            data["Date_Is_Example"] = flag.eq("true")
        else:
            data["Date_Is_Example"] = False
    return data


def filter_sessions(data, start, end, minimum_energy, charger_types):
    """Apply the same filters to every page, chart, KPI and export."""
    mask = (
        data["Session_Date"].between(pd.Timestamp(start), pd.Timestamp(end))
        & (data["Energy_kWh"] >= minimum_energy)
        & data["Charger_Type"].isin(charger_types)
    )
    return data.loc[mask].copy()


def alert_mask(data, temperature):
    """Require BOTH overstay above 20 minutes and temperature above the threshold."""
    return (data["Overstay_Mins"] > 20) & (
        data["Battery_Temp_C"] > temperature
    )


def calculate_kpis(data, temperature):
    """Use supplied cleaned values; changing the alert threshold does not rebill."""
    return [
        data["Energy_kWh"].sum(),
        data["Carbon_Savings_kgCO2"].sum(),
        data["Net_Billable_MYR"].sum(),
        int(alert_mask(data, temperature).sum()),
    ]


def show_kpis(current, previous, temperature, comparable):
    """Show absolute changes against the immediately preceding equal-length period."""
    now = calculate_kpis(current, temperature)
    before = calculate_kpis(previous, temperature)
    labels = [
        "Total Energy Delivered (kWh)",
        "Carbon Offset (kg CO2)",
        "Total Revenue (MYR)",
        "Active Alert Count",
    ]
    for i, (column, label) in enumerate(zip(st.columns(4), labels)):
        value = f"{now[i]:,.0f}" if i == 3 else f"{now[i]:,.2f}"
        if comparable:
            delta = (
                f"{now[i] - before[i]:+,.0f}"
                if i == 3
                else f"{now[i] - before[i]:+,.2f}"
            )
        else:
            delta = None
        column.metric(
            label,
            value,
            delta=delta,
            delta_color="inverse" if i == 3 else "normal",
        )
    st.caption(
        "Revenue means net billable charges, not confirmed receipts. "
        "Carbon offset uses the assessment formula. Active alerts are flagged "
        "records in the selected data, not a live charger feed."
    )


def show_overview(data):
    """Render charts exclusively from the filtered sessions."""
    st.subheader("Filtered charging analytics")
    if data.empty:
        st.info("No sessions match the selected filters.")
        return
    daily_energy = data.groupby("Session_Date")["Energy_kWh"].sum()
    st.caption(
        "Energy by assigned date; example dates do not show a real historical trend."
    )
    st.line_chart(daily_energy.rename("Energy (kWh)"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    sns.boxplot(
        data=data,
        x="User_Type",
        y="Net_Billable_MYR",
        order=["Student", "Staff", "Visitor"],
        color="#9AC9DF",
        ax=axes[0],
    )
    axes[0].set(
        title="Session costs", xlabel="User type", ylabel="Net bill (MYR)"
    )
    sns.scatterplot(
        data=data,
        x="Battery_Temp_C",
        y="Energy_kWh",
        hue="Charger_Type",
        palette={"AC": "#0072B2", "DC": "#D55E00"},
        alpha=0.55,
        ax=axes[1],
    )
    axes[1].set(
        title="Temperature and energy",
        xlabel="Battery temperature (C)",
        ylabel="Energy (kWh)",
    )
    st.pyplot(fig)
    plt.close(fig)


def show_alerts(data, temperature):
    """Display dynamic warning and error banners and the affected records."""
    st.subheader("High-Risk Overstay & Alert Center")
    st.write(
        f"Alert rule: overstay > 20 minutes AND temperature > {temperature} C."
    )
    if data.empty:
        st.info("No sessions match the selected filters.")
        return
    alerts = data.loc[alert_mask(data, temperature)].copy()
    # The additional 5 C margin is a demonstration severity rule, not a safety standard.
    critical = alerts["Battery_Temp_C"] > temperature + 5
    st.caption(
        "Demo severity rule: critical alerts exceed the selected threshold by "
        "more than 5 C. This is not an official battery safety standard."
    )
    if critical.any():
        st.error(f"{int(critical.sum())} critical session(s) require review.")
    if (~critical).any():
        st.warning(
            f"{int((~critical).sum())} additional high-risk session(s) require review."
        )
    if alerts.empty:
        st.success("No selected sessions meet both alert conditions.")
    st.dataframe(alerts, hide_index=True, width="stretch")
    st.caption(
        "Moving the temperature slider changes alerts and alert counts; billed amounts stay unchanged."
    )


def show_data(data, temperature):
    """Display and download the same filtered records, including date provenance."""
    st.subheader("Filtered Data & CSV Export")
    export_data = data.copy()
    export_data["Current_Alert"] = alert_mask(export_data, temperature)
    export_data["Alert_Threshold_C"] = temperature
    st.write(f"Selected records: {len(export_data)}")
    st.dataframe(export_data, hide_index=True, width="stretch")
    st.download_button(
        label="Download filtered CSV",
        data=export_data.to_csv(index=False, date_format="%Y-%m-%d").encode(
            "utf-8-sig"
        ),
        file_name="filtered_energy_data.csv",
        mime="text/csv",
    )
    st.caption(
        "Date_Is_Example is retained in the download so demonstration dates remain identifiable."
    )


def main():
    """Build shared controls and route to three Streamlit pages."""
    st.set_page_config(
        page_title="Campus Energy Dashboard", page_icon="⚡", layout="wide"
    )
    file_path = (
        Path(__file__).resolve().parent
        / "outputs"
        / "cleaned_energy_sales_data.csv"
    )
    try:
        # Cache by file contents so replacing the CSV cannot reuse stale data.
        data = load_data(file_path.read_bytes())
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Cannot load the cleaned dataset: {error}")
        st.stop()

    st.title("Smart Campus Energy & EV Dashboard")
    if data["Date_Is_Example"].any():
        st.warning(
            "EXAMPLE DATES: calendar dates were added only to demonstrate date "
            "filters and KPI comparisons. Energy, fees and temperatures come "
            "from the supplied cleaned data. Date-based results are not real historical findings."
        )
    st.sidebar.header("Shared filters")
    first = data["Session_Date"].min().date()
    last = data["Session_Date"].max().date()
    dates = st.sidebar.date_input(
        "Date range",
        value=(max(first, last - timedelta(days=6)), last),
        min_value=first,
        max_value=last,
    )
    if len(dates) != 2:
        st.info("Select both a start date and an end date.")
        st.stop()
    start, end = dates
    minimum_energy = st.sidebar.slider(
        "Minimum energy (kWh)",
        0.0,
        max(1.0, float(np.ceil(data["Energy_kWh"].max()))),
        0.0,
        step=0.5,
    )
    chargers = st.sidebar.multiselect(
        "Charger Type", ["AC", "DC"], default=["AC", "DC"]
    )
    temperature = st.sidebar.slider(
        "Alert temperature threshold (C)", 30, 60, 40
    )
    current = filter_sessions(data, start, end, minimum_energy, chargers)

    # Compare the selected date range with the immediately preceding range
    # of the same length, using identical energy, charger and alert settings.
    period_days = (end - start).days + 1
    previous_start = start - timedelta(days=period_days)
    previous_end = start - timedelta(days=1)
    previous = filter_sessions(
        data, previous_start, previous_end, minimum_energy, chargers
    )
    comparable = (
        previous_start >= first and not previous.empty and not current.empty
    )
    show_kpis(current, previous, temperature, comparable)
    st.caption(
        f"Selected period: {start} to {end} | Matching sessions: {len(current)}"
    )
    if comparable:
        st.caption(
            f"Delta = selected minus {previous_start} to {previous_end}; absolute changes, not percentages."
        )
    else:
        st.caption(
            "Delta unavailable: the current/prior selection is empty or the prior period extends before the dataset."
        )

    # The entrypoint owns all widgets, so the same filters apply across pages.
    # Each page receives the already filtered DataFrame.
    pages = [
        st.Page(
            lambda: show_overview(current),
            title="Overview",
            url_path="overview",
            default=True,
        ),
        st.Page(
            lambda: show_alerts(current, temperature),
            title="Alert Center",
            url_path="alerts",
        ),
        st.Page(
            lambda: show_data(current, temperature),
            title="Data & Export",
            url_path="data",
        ),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
