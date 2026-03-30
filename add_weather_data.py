"""Weather data collection and integration utilities.

This module fetches historical weather data from the Open-Meteo API
and merges it with carbon intensity datasets. Weather features
(wind speed, solar radiation, cloud cover, etc.) are key predictors
for grid carbon intensity because renewable energy output depends
heavily on atmospheric conditions.
"""

import glob  # File pattern matching to find carbon CSVs
from datetime import datetime  # Date handling for API parameters
from pathlib import Path

import pandas as pd  # DataFrame manipulation and CSV I/O
import requests  # HTTP client for Open-Meteo API calls
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LAG_STEPS = [1, 2, 3, 6, 12, 24, 48]
ROLLING_WINDOWS = [3, 6, 12, 24]
LAG_COLUMNS = []
ROLLING_COLUMNS = []


def _to_utc_naive(series: pd.Series) -> pd.Series:
    """Parse timestamps as UTC and return timezone-naive UTC timestamps."""
    return pd.to_datetime(series, utc=True).dt.tz_convert(None)


def add_lag_features(df: pd.DataFrame, columns: list[str], lag_steps: list[int]) -> pd.DataFrame:
    """Add lagged features for selected columns in timestamp order."""
    lagged_df = df.sort_values("timestamp").copy()
    available_columns = [col for col in columns if col in lagged_df.columns]

    for column in available_columns:
        for lag in lag_steps:
            new_col = f"{column}_lag_{lag}"
            if new_col not in lagged_df.columns:
                lagged_df[new_col] = lagged_df[column].shift(lag)

    return lagged_df


def add_rolling_mean_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Add past-only rolling mean features for selected columns."""
    rolling_df = df.sort_values("timestamp").copy()
    available_columns = [col for col in columns if col in rolling_df.columns]

    for column in available_columns:
        for window in windows:
            new_col = f"{column}_rolling_mean_{window}"
            if new_col not in rolling_df.columns:
                rolling_df[new_col] = rolling_df[column].shift(1).rolling(window=window).mean()

    return rolling_df


class WeatherDataCollector:
    """Fetches historical weather data from Open-Meteo.

    Open-Meteo is a free weather API that provides hourly historical
    data globally without requiring an API key. We use it to collect
    variables relevant for predicting renewable energy generation.
    """

    def __init__(self, latitude=51.5074, longitude=-0.1278):
        """Initialise the collector with a geographic location.

        Args:
            latitude (float): Latitude of the target location (default: London).
            longitude (float): Longitude of the target location (default: London).
        """
        # Store location for all subsequent API requests
        self.latitude = latitude
        self.longitude = longitude
        # Open-Meteo archive endpoint for historical data
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        # Reusable session with retry for transient network/API failures.
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def fetch_historical_weather(self, start_date, end_date):
        """Fetch hourly historical weather data from Open-Meteo.

        Args:
            start_date: Start of the date range (datetime or 'YYYY-MM-DD').
            end_date: End of the date range (datetime or 'YYYY-MM-DD').

        Returns:
            pandas.DataFrame | None: Hourly weather records, or None on failure.
        """
        # Normalise dates to string format expected by the API
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%d")

        print(f"Fetching weather data from {start_date} to {end_date}...")
        print(f"Location: ({self.latitude}, {self.longitude})")

        # Request parameters including all weather variables that
        # are useful for predicting renewable energy output.
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m",  # Air temperature at 2 m height (°C)
                "windspeed_10m",  # Wind speed at 10 m — affects small turbines
                "windspeed_100m",  # Wind speed at 100 m — typical turbine hub height
                "winddirection_10m",  # Wind direction in degrees
                "cloudcover",  # Cloud cover percentage (solar predictor)
                "shortwave_radiation",  # Incoming solar radiation (W/m²)
                "precipitation",  # Precipitation (mm)
                "surface_pressure",  # Atmospheric pressure (hPa)
                "relativehumidity_2m",  # Relative humidity (%)
            ],
            "timezone": "GMT",  # Align with UK carbon intensity timestamps
        }

        try:
            # Make the HTTP GET request to Open-Meteo
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()  # Raise for HTTP errors (4xx/5xx)
            data = response.json()

            # The API returns hourly data keyed under 'hourly'
            hourly_data = data["hourly"]

            required_keys = {
                "time",
                "temperature_2m",
                "windspeed_10m",
                "windspeed_100m",
                "winddirection_10m",
                "cloudcover",
                "shortwave_radiation",
                "precipitation",
                "surface_pressure",
                "relativehumidity_2m",
            }
            missing_keys = sorted(required_keys.difference(hourly_data.keys()))
            if missing_keys:
                raise KeyError(f"Missing expected weather keys: {missing_keys}")

            # Build a tidy DataFrame with renamed columns for clarity
            df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(hourly_data["time"]),
                    "temperature": hourly_data["temperature_2m"],
                    "wind_speed_10m": hourly_data["windspeed_10m"],
                    "wind_speed_100m": hourly_data["windspeed_100m"],
                    "wind_direction": hourly_data["winddirection_10m"],
                    "cloud_cover": hourly_data["cloudcover"],
                    "solar_radiation": hourly_data["shortwave_radiation"],
                    "precipitation": hourly_data["precipitation"],
                    "pressure": hourly_data["surface_pressure"],
                    "humidity": hourly_data["relativehumidity_2m"],
                }
            )

            print(f"✓ Fetched {len(df)} hourly weather records")
            return df

        except requests.RequestException as e:
            print(f"✗ Network/API error fetching weather data: {e}")
            return None
        except (KeyError, ValueError) as e:
            print(f"✗ Unexpected weather payload format: {e}")
            return None
        except TypeError as e:
            print(f"✗ Invalid weather payload type: {e}")
            return None


def merge_weather_with_carbon(carbon_file, weather_df, output_file=None):
    """Merge weather data with a carbon intensity CSV on timestamp.

    This produces a single DataFrame containing both carbon intensity
    readings and the corresponding weather conditions, which can then
    be used for analysis or ML model training.

    Args:
        carbon_file (str): Path to the carbon intensity CSV.
        weather_df (pandas.DataFrame): DataFrame with hourly weather data.
        output_file (str | None): Destination CSV path; auto-generated if None.

    Returns:
        pandas.DataFrame: Merged dataset with weather features appended.
    """

    # Load the carbon intensity data from disk
    carbon_df = pd.read_csv(carbon_file)
    carbon_df["timestamp"] = _to_utc_naive(carbon_df["timestamp"])
    weather_df = weather_df.copy()
    weather_df["timestamp"] = _to_utc_naive(weather_df["timestamp"])

    # Log basic stats to help with debugging mismatches
    print(f"Carbon data: {len(carbon_df)} records")
    print(f"  Date range: {carbon_df['timestamp'].min()} to {carbon_df['timestamp'].max()}")
    print(f"Weather data: {len(weather_df)} records")
    print(f"  Date range: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")

    # As-of merge is more robust than exact timestamp joins when cadences differ.
    merged = pd.merge_asof(
        carbon_df.sort_values("timestamp"),
        weather_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",  # only use past weather
        tolerance=pd.Timedelta("59min"),  # within the same hour
    )
    merged_df = merged.copy()

    # Warn if any weather columns are still missing values (unlikely but possible)
    missing_weather = merged_df["temperature"].isna().sum()
    if missing_weather > 0:
        print(f"⚠ Warning: {missing_weather} timestamps missing weather data")
        merged_df = merged_df.dropna()
        print(f"  Dropped rows with missing data. New size: {len(merged_df)}")

    merged_df = add_lag_features(merged_df, LAG_COLUMNS, LAG_STEPS)
    merged_df = add_rolling_mean_features(merged_df, ROLLING_COLUMNS, ROLLING_WINDOWS)

    # Lag and rolling features create NaNs at the top of the series by design.
    rows_before_temporal_drop = len(merged_df)
    merged_df = merged_df.dropna()
    rows_dropped_temporal = rows_before_temporal_drop - len(merged_df)
    if rows_dropped_temporal > 0:
        print(
            f"  Dropped {rows_dropped_temporal} initial rows after lag/rolling feature generation."
        )

    print(f"\n✓ Merged dataset: {len(merged_df)} records")
    print(f"  Total features: {len(merged_df.columns)}")

    # Auto-generate output filename if not provided
    if output_file is None:
        output_file = carbon_file.replace(".csv", "_with_weather.csv")

    # Persist the enriched dataset
    merged_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved merged data to: {output_file}")

    return merged_df


if __name__ == "__main__":
    # -------------------------------------------------------------------
    # Entry point: find the most recent carbon CSV, fetch matching
    # weather data, and produce a merged dataset.
    # -------------------------------------------------------------------

    # Locate any existing carbon intensity CSVs
    carbon_files = [
        path
        for path in glob.glob("carbon_data/uk_carbon_intensity_*.csv")
        if "_with_weather.csv" not in path
    ]
    carbon_files = sorted(carbon_files)

    if not carbon_files:
        # Fail loudly if the prerequisite data is missing
        raise FileNotFoundError("No carbon CSV files found in carbon_data/")
    else:
        collector = WeatherDataCollector(latitude=51.5074, longitude=-0.1278)
        for carbon_file in carbon_files:
            try:
                print(f"Found carbon data file: {carbon_file}")

                # Read the carbon data to determine the date range needed for weather
                carbon_df = pd.read_csv(carbon_file)
                carbon_df["timestamp"] = pd.to_datetime(carbon_df["timestamp"])
                start_date = carbon_df["timestamp"].min()
                end_date = carbon_df["timestamp"].max()

                print(f"\nDate range: {start_date.date()} to {end_date.date()}")
                print("\nUsing location: London (51.5074, -0.1278)")

                # Create a collector for London and fetch weather over the same period
                weather_df = collector.fetch_historical_weather(start_date, end_date)

                if weather_df is not None:
                    # Combine weather features with carbon intensity data
                    output_file = str(Path(carbon_file).with_suffix("")) + "_with_weather.csv"
                    _ = merge_weather_with_carbon(carbon_file, weather_df, output_file=output_file)
            except Exception as e:
                print(f"Error processing {carbon_file}: {e}")
                continue
