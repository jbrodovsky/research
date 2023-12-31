"""
Toolbox for processing raw data collected from the SensorLogger app and the MGD77T format.
"""

import os
import argparse
import fnmatch
import pytz
import pandas as pd

from datetime import timedelta


# Sensor Logger Processing
def process_sensor_logger_dataset(folder: str):
    """
    Process the raw .csv files recorded from the SensorLogger app. This function will correct
    and rectify the coordinate frame as well as rename the recorded variables.

    Parameters
    ----------
    :param folder: the filepath to the folder containing the raw data values. This folder should
    contain TotalAcceleration.csv, Gyroscope.csv, Magnetometer.csv, Barometer.csv, and
    LocationGps.csv
    :type folder: string

    :returns: Pandas dataframes corresponding to the processed and cleaned imu, magnetometer,
    barometer, and GPS data.
    """

    accel = pd.read_csv(
        f"{folder}/TotalAcceleration.csv", sep=",", header=0, index_col=0, dtype=float
    )
    accel = accel.rename(columns={"z": "a_z", "y": "a_y", "x": "a_x"})
    accel["a_z"] = -accel["a_z"]
    accel = accel.drop(columns="seconds_elapsed")

    gyros = pd.read_csv(
        f"{folder}/Gyroscope.csv", sep=",", header=0, index_col=0, dtype=float
    )
    gyros["y"] = -gyros["y"]
    gyros = gyros.rename(columns={"z": "w_z", "y": "w_y", "x": "w_x"})
    gyros = gyros.drop(columns="seconds_elapsed")

    imu = accel.merge(gyros, how="outer", left_index=True, right_index=True)
    imu = imu.fillna(value=pd.NA)
    imu = _convert_datetime(imu)

    magnetometer = pd.read_csv(
        f"{folder}/Magnetometer.csv", sep=",", header=0, index_col=0, dtype=float
    )
    magnetometer["z"] = -magnetometer["z"]
    magnetometer = magnetometer.rename(
        columns={"z": "mag_z", "y": "mag_y", "x": "mag_x"}
    )
    magnetometer = magnetometer.drop(columns="seconds_elapsed")
    magnetometer = _convert_datetime(magnetometer)

    barometer = pd.read_csv(
        f"{folder}/Barometer.csv", sep=",", header=0, index_col=0, dtype=float
    )
    barometer = barometer.drop(columns="seconds_elapsed")
    barometer = _convert_datetime(barometer)

    gps = pd.read_csv(
        f"{folder}/LocationGps.csv", sep=",", header=0, index_col=0, dtype=float
    )
    gps = gps.drop(columns="seconds_elapsed")
    gps = _convert_datetime(gps)

    return imu, magnetometer, barometer, gps


def save_sensor_logger_dataset(
    imu: pd.DataFrame,
    magnetometer: pd.DataFrame,
    barometer: pd.DataFrame,
    gps: pd.DataFrame,
    output_format: str = "csv",
    output_folder: str = "./",
) -> None:
    """
    Saves the processed sensor logger data. Data is saved to a folder.

    Parameters
    ----------
    :param imu: IMU data.
    :type imu: pandas.DataFrame
    :param magnetometer: Magnetometer data.
    :type magnetometer: pandas.DataFrame
    :param barometer: barometer data.
    :type barometer: pandas.DataFrame
    :param gps: GPS data.
    :type gps: pandas.DataFrame
    :param output_format: file extension and format for output files. Optional.
    :type output_format: string
    :output_folder: filepath and/or folder name for output. Optional.
    :type output_folder: string

    :returns: none
    """
    if output_format == "csv":
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        imu.to_csv(f"{output_folder}/imu.csv")
        magnetometer.to_csv(f"{output_folder}/magnetometer.csv")
        barometer.to_csv(f"{output_folder}/barometer.csv")
        gps.to_csv(f"{output_folder}/gps.csv")
    else:
        print("Other file formats not implemented yet.")


def _convert_datetime(
    df: pd.DataFrame, timezone: str = "America/New_York"
) -> pd.DataFrame:
    """ """
    dates = pd.to_datetime(df.index / 1e9, unit="s").tz_localize("UTC")
    df.index = dates.tz_convert(pytz.timezone(timezone))
    df = df.resample("1s").mean()
    return df


def process_sensorlogger(args):
    """
    Process the raw .csv files recorded from the SensorLogger app. This function will correct
    and rectify the coordinate frame as well as rename the recorded variables.
    """
    assert os.path.exists(args.location) or os.path.isdir(
        args.location
    ), "Error: invalid location for input data. Please verify file path."
    imu, magnetic_anomaly, barometer, gps = process_sensor_logger_dataset(args.location)
    output_folder = os.path.join(args.output, "processed")
    save_sensor_logger_dataset(output_folder, imu, magnetic_anomaly, barometer, gps)


# MGD77T Processing
def m77t_to_csv(data: pd.DataFrame) -> pd.DataFrame:
    """
    Formats a .m77t file in a Pandas data frame to a more useful representation. Data is read in
    and the time data is foramtted to a Python `datetime` object and used as the new index of the
    DataFrame. Rows containing N/A values are dropped.

    Parameters
    -----------
    :param data: the raw input data from the .m77t read in via a Pandas DataFrame
    :type data: Pandas DataFrame

    :returns: the time indexed and down sampled data frame.
    """
    data = data.dropna(subset=["TIME"])
    # Reformate date, time, and timezone data from dataframe to propoer Python datetime
    dates = data["DATE"].astype(int)
    times = (data["TIME"].astype(float)).apply(int)
    timezones = data["TIMEZONE"].astype(int)
    timezones = timezones.apply(lambda tz: f"+{tz:02}00" if tz >= 0 else f"{tz:02}00")
    times = times.apply(lambda time_int: f"{time_int // 100:02d}{time_int % 100:02d}")
    datetimes = dates.astype(str) + times.astype(str)
    timezones.index = datetimes.index
    datetimes += timezones.astype(str)
    datetimes = pd.to_datetime(datetimes, format="%Y%m%d%H%M%z")
    data.index = datetimes
    # Clean up the rest of the data frame
    data = data.drop(columns=["DATE", "TIME", "TIMEZONE", "SURVEY_ID"])
    data = data.dropna(axis=1, how="all")
    data = data.dropna(axis=0, how="any")
    # Sort the DataFrame by the index
    data = data.sort_index()
    # Remove duplicate index values
    data = data.loc[~data.index.duplicated(keep="last")]

    return data


def _search_folder(folder_path: str, extension: str) -> list:
    """
    Recursively search through a given folder to find files of a given file's
    extension. File extension must be formatted as: .ext with an astericks.

    Parameters
    ----------
    :param folder_path: The file path to the root folder to search.
    :type folder_path: STRING
    :param extension: The file extension formatted as .ext to search for.
    :type extension: STRING

    Returns
    -------
    :returns: list of filepaths to the file types of interest.

    """
    new_file_paths = []
    for root, _, files in os.walk(folder_path):
        print(f"Searching: {root}")
        for filename in fnmatch.filter(files, extension):
            print(f"Adding: {os.path.join(root, filename)}")
            new_file_paths.append(os.path.join(root, filename))
    return new_file_paths


def process_mgd77_dataset(folder_path: str, output_path: str) -> None:
    """
    Recursively search through a given folder to find .m77t files. When found,
    read them into memory using Pandas, processes them, and then save as a
    .csv to the location specified by `output_path`.

    Parameters
    ----------
    :param folder_path: The file path to the root folder to search.
    :type folder_path: STRING
    :param output_path: The file path to save data.
    :type output_path: STRING

    Returns
    -------
    None

    """
    file_paths = _search_folder(folder_path, "*.m77t")
    print("Found the following source files:")
    print("\n".join(file_paths))
    for file_path in file_paths:
        filename = os.path.split(file_path)[-1]
        print(f"Processing: {filename}")
        name = filename.split(".m77t")[0]
        print(f"Saving as: {name}.csv")
        data_in = pd.read_csv(
            file_path,
            sep="\t",
            header=0,
        )
        data_out = m77t_to_csv(data=data_in)
        data_out.to_csv(os.path.join(output_path, f"{name}.csv"))


def process_mgd77(args):
    """
    Recursively search through a given folder to find .m77t files. When found,
    read them into memory using Pandas, processes them, and then save as a
    .csv to the location specified by `output_path`.
    """
    if os.path.isdir(args.location):
        if not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)
        process_mgd77_dataset(args.location, args.output)
    else:
        filename = args.location.split("\\")[-1]
        filename = filename.split(".m77t")[0] + ".csv"
        data = pd.read_csv(args.location, sep="\t", header=0)
        data = m77t_to_csv(data)
        data.to_csv(os.path.join(args.output, filename))


# MGD77T parsing
def parse_dataset(args):
    """
    Recursively search through a given folder to find .csv files. When found,
    read them into memory using parse_trackline, processes them, and then save as a
    .csv to the location specified by `output_path`.
    """
    file_paths = _search_folder(args.location, "*.csv")
    print("Found the following source files:")
    print("\n".join(file_paths))
    for file_path in file_paths:
        filename = os.path.split(file_path)[-1]
        print(f"Processing: {filename}")
        parse_trackline(
            file_path,
            save=True,
            output_dir=args.output,
            max_time=timedelta(minutes=args.max_time),
            max_delta_t=timedelta(minutes=args.max_delta_t),
            min_duration=timedelta(minutes=args.min_duration),
        )
        # data_out.to_csv(f"{output_path}/{name}.csv")
        # data_out.to_csv(os.path.join(args.output, f"{name}.csv"))


def parse_trackline(
    filepath: str,
    max_time: timedelta = timedelta(minutes=10),
    max_delta_t: timedelta = timedelta(minutes=2),
    min_duration: timedelta = timedelta(minutes=60),
    save: bool = False,
    output_dir: str = None,
) -> list:
    """
    Parse a trackline dataset into periods of continuous data.
    """
    data = pd.read_csv(
        filepath,
        header=0,
        index_col=0,
        parse_dates=True,
        dtype={
            "LAT": float,
            "LON": float,
            "BAT_TTIME": float,
            "CORR_DEPTH": float,
            "MAG_TOT": float,
            "MAG_RES": float,
        },
    )
    # get the filename without the extension
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    # Split the dataset into periods of continuous data
    data["DT"] = data.index.to_series().diff()
    data.loc[data.index[0], "DT"] = timedelta(seconds=0)
    inds = (data["DT"] > max_time).to_list()
    subsets = find_periods(inds)
    subsections = split_dataset(data, subsets)
    # Validate the subsections
    validated_subsections = []
    for i, df in enumerate(subsections):
        # Check that the time between each data point is less than the max delta t
        if not (df["DT"] < max_delta_t).mean():
            # print(f"Subsection {i} of {file_name} failed validation")
            continue
        # Check that the subsection meets the minimum duration
        if len(df) < 3 or df.index[-1] - df.index[0] < min_duration:
            # print(f"Subsection {i} of {file_name} failed validation")
            continue
        # Check that the subsection has at least 2 unique timestamps
        if len(df.index.unique()) < 2:
            # print(f"Subsection {i} of {file_name} failed validation")
            continue
        validated_subsections.append(df)
    # Save off the subsections to CSV files
    if save:
        if output_dir is not None and not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if output_dir is None:
            output_dir = ""

        for i, df in enumerate(validated_subsections):
            df.to_csv(os.path.join(output_dir, f"{file_name}_{i}.csv"))

    return subsections


def find_periods(mask) -> list:
    """
    Find the start and stop indecies from a boolean mask.
    """
    # Calculate the starting and ending indices for each period
    periods = []
    start_index = None

    for idx, is_true in enumerate(mask):
        if not is_true and start_index is None:
            start_index = idx
        elif is_true and start_index is not None:
            end_index = idx - 1
            periods.append((start_index, end_index))
            start_index = None

    # If the last period extends until the end of the mask, add it
    if start_index is not None:
        end_index = len(mask) - 1
        periods.append((start_index, end_index))

    return periods


def split_dataset(df: pd.DataFrame, periods: list) -> list:
    """
    Split a dataframe into subsections based on the given periods.
    """
    subsections = []
    for start, end in periods:
        subsection = df.iloc[start : end + 1]  # Add 1 to include the end index
        subsections.append(subsection)
    return subsections


# Command Line Interface
def parse_args():
    """
    Command line interface argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="SensorLoggerProcessor",
        description="Post-process the raw datasets collected by the Sensor Logger App",
    )

    parser.add_argument(
        "--mode",
        choices=["sensorlogger", "mgd77", "parser"],
        required=True,
        help="Type of sensor recording to process. Parser is used to parse NOAA datasets in the mgd77t format into continuous datasets.",
    )
    parser.add_argument(
        "--location",
        default="./",
        help="Path to the data. Can either be a direct file path to the .m77t file, "
        + "a folder containing such file(s), or the folder containing the raw .csvs from "
        + "the sensor logger. If a folder is given, each subfolder is searched for files.",
        required=True,
    )
    parser.add_argument(
        "--output",
        default="./",
        help="Output filepath to save off processed data",
        required=False,
    )
    parser.add_argument(
        "--format",
        choices=["csv"],
        default="csv",
        required=False,
        help="Output format for processed data. Default is .csv; other options to be "
        + "implemented later .db, .h5",
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=10,
        required=False,
        help="Maximum time between data points to be considered a continuous dataset. "
        + "Default is 10 minutes. Input units are minutes.",
    )
    parser.add_argument(
        "--max_delta_t",
        type=float,
        default=2,
        required=False,
        help="Maximum time between data points to be considered a continuous dataset. "
        + "Default is 2 minutes. Input units are minutes.",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=60,
        required=False,
        help="Minimum duration of a continuous dataset. Default is 60 minutes. Input "
        + "units are minutes.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Command line interface for processing the raw datasets collected by the Sensor Logger App or
    NOAA datasets in the mgd77t format.
    """
    args = parse_args()

    process_map = {
        "sensorlogger": process_sensorlogger,
        "mgd77": process_mgd77,
        "parser": parse_dataset,
    }
    if args.mode in process_map:
        process_map[args.mode](args)
    else:
        raise NotImplementedError(
            f"Parser mode type {args.mode} not recognized. Please choose from the following: sensorlogger, mgd77, parser"
        )


if __name__ == "__main__":
    main()
