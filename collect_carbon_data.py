"""Carbon intensity data collection utilities.

This module fetches historical carbon intensity data from the
UK National Grid (carbonintensity.org.uk) and prepares it for
analysis or downstream optimisation (e.g., EV charging schedules).

The implemented collector handles API chunking, basic parsing,
and writing a CSV of cleaned, time-featured records.
"""

import os  # filesystem operations
import time  # for polite sleep between requests
from datetime import datetime, timedelta  # date arithmetic

import pandas as pd  # Data manipulation
import requests  # HTTP client for calling the public API

LAG_STEPS = [1, 2, 3, 6, 12, 24, 48]  # in 30-min units
# lag_1 = 30min, lag_2 = 1hr, lag_48 = 24hr
LAG_COLUMNS = ["carbon_intensity"]
ROLLING_WINDOWS = [3, 6, 12, 24]
ROLLING_COLUMNS = ["carbon_intensity"]


def add_lag_features(df: pd.DataFrame, columns: list[str], lag_steps: list[int]) -> pd.DataFrame:
    """Add simple lagged versions of selected columns to a time-ordered dataframe."""
    lagged_df = df.sort_values("timestamp").copy()
    for column in columns:
        for lag in lag_steps:
            lagged_df[f"{column}_lag_{lag}"] = lagged_df[column].shift(lag)
    return lagged_df


def add_rolling_mean_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Add past-only rolling mean features to a time-ordered dataframe."""
    rolling_df = df.sort_values("timestamp").copy()
    for column in columns:
        for window in windows:
            rolling_df[f"{column}_rolling_mean_{window}"] = (
                rolling_df[column].shift(1).rolling(window=window).mean()
            )
    return rolling_df


class CarbonDataCollector:
    """Collects carbon intensity data from various sources.

    Currently implemented:
    - `collect_uk_national_grid`: pulls historical data from
      https://api.carbonintensity.org.uk/intensity
    """

    def __init__(self):
        """Initialise the collector and ensure the data directory exists."""
        # Directory where CSV outputs will be written
        self.data_dir = "carbon_data"
        os.makedirs(self.data_dir, exist_ok=True)

    def collect_uk_national_grid(self, days_back=30):
        """Fetch UK carbon intensity data for the past `days_back` days.

        The National Grid API restricts long requests, so we fetch data
        in chunks (default chunk size is 14 days) and combine the results.

        Args:
            days_back (int): how many days of historical data to fetch.

        Returns:
            pandas.DataFrame | None: cleaned DataFrame on success, otherwise None.
        """
        print(f"Collecting UK National Grid data for last {days_back} days...")

        # Compute the date range for requests
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        base_url = "https://api.carbonintensity.org.uk/intensity"

        all_data = []  # accumulator for JSON objects returned by the API

        # The API allows shorter ranges; fetch in chunks to avoid errors.
        current_date = start_date
        chunk_size = 14  # days per chunk (keeps requests reasonably-sized)

        while current_date < end_date:
            # chunk_end is the smaller of the desired end or the chunk boundary
            chunk_end = min(current_date + timedelta(days=chunk_size), end_date)

            # API expects ISO-like timestamps without seconds: YYYY-MM-DDTHH:MMZ
            from_date = current_date.strftime("%Y-%m-%dT%H:%MZ")
            to_date = chunk_end.strftime("%Y-%m-%dT%H:%MZ")

            chunk_url = f"{base_url}/{from_date}/{to_date}"

            try:
                # Perform the HTTP GET and raise for non-2xx responses
                response = requests.get(chunk_url)
                response.raise_for_status()
                data = response.json()

                # The API returns a top-level 'data' list containing records
                if "data" in data:
                    all_data.extend(data["data"])
                    print(
                        f"Collected {len(data['data'])} records "
                        f"({current_date.date()} to {chunk_end.date()})"
                    )

                # Small sleep to avoid hammering the public API
                time.sleep(0.5)

            except Exception as e:
                # Log the error and continue with the next chunk
                print(f"  ✗ Error collecting chunk: {e}")

            # advance to the next chunk
            current_date = chunk_end

        # Convert the accumulated list of JSON records into a DataFrame
        df = pd.DataFrame(all_data)

        if not df.empty:
            # Parse and normalise fields from the API response
            # 'from' is the start timestamp for the interval
            df["timestamp"] = pd.to_datetime(df["from"])

            # 'intensity' may be a dict with 'actual' and 'forecast' values,
            # so safely extract those fields when present.
            df["carbon_intensity"] = df["intensity"].apply(
                lambda x: x.get("actual") if isinstance(x, dict) else None
            )
            df["forecast"] = df["intensity"].apply(
                lambda x: x.get("forecast") if isinstance(x, dict) else None
            )
            # Add convenient time-derived features for analysis
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["month"] = df["timestamp"].dt.month
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

            # Add lag features so downstream models can see recent history.
            df = add_lag_features(df.sort_values("timestamp"), LAG_COLUMNS, LAG_STEPS)
            df = add_rolling_mean_features(df, ROLLING_COLUMNS, ROLLING_WINDOWS)

            # Keep only the columns we need and drop rows with missing values
            df = df[
                [
                    "timestamp",
                    "carbon_intensity",
                    "forecast",
                    "hour",
                    "day_of_week",
                    "month",
                    "is_weekend",
                    *[f"{column}_lag_{lag}" for column in LAG_COLUMNS for lag in LAG_STEPS],
                    *[
                        f"{column}_rolling_mean_{window}"
                        for column in ROLLING_COLUMNS
                        for window in ROLLING_WINDOWS
                    ],
                ]
            ]
            df = df.dropna()

            # Persist the cleaned dataset for later use
            filename = (
                f"{self.data_dir}/uk_carbon_intensity_{start_date.date()}_to_{end_date.date()}.csv"
            )
            df.to_csv(filename, index=False)

            # Print summary information to help with quick sanity checks
            print(f"\n✓ Saved {len(df)} records to {filename}")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(
                f"Carbon intensity range: "
                f"{df['carbon_intensity'].min():.1f} - {df['carbon_intensity'].max():.1f} gCO2/kWh"
            )
            return df

        else:
            print("No data collected")
            return None


def collect_data_uk():
    collector = CarbonDataCollector()
    df = collector.collect_uk_national_grid(
        days_back=1095 * 5
    )  # Collect 2 years of data (1095 days)

    if df is not None:
        print(f"Total records: {len(df)}")
        print("\nCarbon intensity statistics (gCO2/kWh):")
        print(df["carbon_intensity"].describe())
        print("\nHourly patterns (average carbon intensity by hour):")
        print(df.groupby("hour")["carbon_intensity"].mean().round(1))

    return df


if __name__ == "__main__":
    df = collect_data_uk()
    print("\n✓ Data collection complete!")
