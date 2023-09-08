"""
This module is used to pre-process MGD77 data from NOAA into a simplified comma separated value format that can easily be read by Pandas.
"""

import pandas as pd
import os, argparse, fnmatch


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
    TZs = data["TIMEZONE"].astype(int)
    TZs = TZs.apply(lambda tz: f"+{tz:02}00" if tz >= 0 else f"{tz:02}00")

    times = times.apply(lambda time_int: f"{time_int // 100:02d}{time_int % 100:02d}")
    datetimes = dates.astype(str) + times.astype(str)

    TZs.index = datetimes.index
    datetimes += TZs.astype(str)

    DateTimes = pd.to_datetime(datetimes, format="%Y%m%d%H%M%z")

    data.index = DateTimes
    data = data.drop(columns=["DATE", "TIME", "TIMEZONE"])
    data = data.dropna(axis=1, how="all")
    data = data.dropna(axis=0, how="any")

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
        # for subfolder in subfolders:
        #    new_file_paths.append(search_folder(subfolder, extension))
    return new_file_paths


def preprocess_data(folder_path: str, output_path: str) -> None:
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
    print
    for file_path in file_paths:
        filename = file_path.split("\\")[-1]
        filename = filename.split(".m77t")[0]
        data_in = pd.read_csv(
            file_path,
            sep="\t",
            header=0,
        )
        data_out = m77t_to_csv(data=data_in)
        data_out.to_csv(f"{output_path}/{filename}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MGD77DataProcessor",
        description="Pre-processor for NOAA MGD77 data format (.m77t)",
    )
    parser.add_argument(
        "--location",
        default="./",
        help="Path to the data. Can either be a direct file path to the .m77t file or to a folder containing such file(s). If a folder is given, each subfolder is searched for .m77t files.",
        required=True,
    )
    parser.add_argument(
        "--output",
        default="./",
        help="Output filepath to save off processed data",
        required=False,
    )

    args = parser.parse_args()

    assert os.path.exists(args.location) or os.path.isdir(
        args.location
    ), "Error: invalid location for input data. Please verify file path."

    if os.path.isdir(args.location):
        if not os.path.isdir(args.output):
            os.mkdir(args.output)
        preprocess_data(args.location, args.output)
    else:
        filename = args.location.split("\\")[-1]
        filename = filename.split(".m77t")[0] + ".csv"
        data = pd.read_csv(args.location, sep="\t", header=0)
        data = m77t_to_csv(data)
        data.to_csv(os.path.join(args.output, filename))
