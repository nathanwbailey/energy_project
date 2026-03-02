"""Weather data collection and integration utilities.

This module fetches historical weather data from the Open-Meteo API
and merges it with carbon intensity datasets. Weather features
(wind speed, solar radiation, cloud cover, etc.) are key predictors
for grid carbon intensity because renewable energy output depends
heavily on atmospheric conditions.
"""

import glob  # File pattern matching to find carbon CSVs
from datetime import datetime  # Date handling for API parameters

import pandas as pd  # DataFrame manipulation and CSV I/O
import requests  # HTTP client for Open-Meteo API calls


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
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise for HTTP errors (4xx/5xx)
            data = response.json()

            # The API returns hourly data keyed under 'hourly'
            hourly_data = data["hourly"]

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

        except Exception as e:
            # Log the error and return None so callers can handle gracefully
            print(f"✗ Error fetching weather data: {e}")
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
    carbon_df["timestamp"] = pd.to_datetime(carbon_df["timestamp"])

    # Remove timezone info so both DataFrames can merge cleanly
    carbon_df["timestamp"] = carbon_df["timestamp"].dt.tz_localize(None)
    weather_df["timestamp"] = weather_df["timestamp"].dt.tz_localize(None)

    # Log basic stats to help with debugging mismatches
    print(f"Carbon data: {len(carbon_df)} records")
    print(f"  Date range: {carbon_df['timestamp'].min()} to {carbon_df['timestamp'].max()}")
    print(f"Weather data: {len(weather_df)} records")
    print(f"  Date range: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")

    # Inner merge keeps only timestamps present in both datasets
    merged = pd.merge(carbon_df, weather_df, on="timestamp")
    merged_df = merged.copy()

    # Warn if any weather columns are still missing values (unlikely but possible)
    missing_weather = merged_df["temperature"].isna().sum()
    if missing_weather > 0:
        print(f"⚠ Warning: {missing_weather} timestamps missing weather data")
        merged_df = merged_df.dropna()
        print(f"  Dropped rows with missing data. New size: {len(merged_df)}")

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
    carbon_files = glob.glob("carbon_data/uk_carbon_intensity_*.csv")

    if not carbon_files:
        # Fail loudly if the prerequisite data is missing
        raise FileNotFoundError("No carbon CSV files found in carbon_data/")
    else:
        # Use the most recent file (alphabetically last after sorting by date)
        latest_carbon_file = sorted(carbon_files)[-1]
        print(f"\nFound carbon data: {latest_carbon_file}")

        # Read the carbon data to determine the date range needed for weather
        carbon_df = pd.read_csv(latest_carbon_file)
        carbon_df["timestamp"] = pd.to_datetime(carbon_df["timestamp"])
        start_date = carbon_df["timestamp"].min()
        end_date = carbon_df["timestamp"].max()

        print(f"\nDate range: {start_date.date()} to {end_date.date()}")
        print("\nUsing location: London (51.5074, -0.1278)")

        # Create a collector for London and fetch weather over the same period
        collector = WeatherDataCollector(latitude=51.5074, longitude=-0.1278)
        weather_df = collector.fetch_historical_weather(start_date, end_date)

        if weather_df is not None:
            # Combine weather features with carbon intensity data
            merged_df = merge_weather_with_carbon(latest_carbon_file, weather_df)
