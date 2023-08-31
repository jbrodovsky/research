import pandas as pd
import pytz, os, argparse


def process_dataset(folder: str):
    """ """

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


def save_dataset(
    filepath: str,
    imu: pd.DataFrame,
    magnetometer: pd.DataFrame,
    barometer: pd.DataFrame,
    gps: pd.DataFrame,
    output_format: str = "csv",
):
    """ """
    if output_format == "csv":
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="SensorLoggerProcessor",
        description="Post-process the raw datasets collected by the Sensor Logger App",
    )
    parser.add_argument("--folder", default="./", help="Folder containing the dataset")
    parser.add_argument(
        "--output",
        choices=["csv"],
        default="csv",
        required=False,
        help="Output format for processed data. Default is .csv; other options to be implemented later .db, .h5",
    )
    args = parser.parse_args()

    assert os.path.exists(args.folder) and os.path.isdir(
        args.folder
    ), "ERROR: dataset folder not found."

    IMU, MAG, BAR, GPS = process_dataset(args.folder)
    output_folder = f"{args.folder}/processed"
    save_dataset(output_folder, IMU, MAG, BAR, GPS)
