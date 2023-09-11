"""
Python module to access and save geophysical maps using GMT.
"""

import subprocess
from argparse import ArgumentParser
from tools import wrap_to_180
import xarray as xr


def get_map_section(
    west_lon: float,
    east_lon: float,
    south_lat: float,
    north_lat: float,
    map_type: str = "relief",
    map_res: str = "02m",
    output_location: str = "./",
) -> xr.DataArray:
    """
    Function for querying the raw map to get map segments. This is the publicly facing function
    that should be used in other modules for reading in maps from raw data. If you don't need to
    use raw data, use a load_map_file().
    """
    _get_map_section(
        west_lon, east_lon, south_lat, north_lat, map_type, map_res, output_location
    )
    out = load_map_file(output_location)
    return out


def load_map_file(filepath: str) -> xr.DataArray:
    """
    Used to load the local .nc (netCDF) map files in to a Python xarray DataArray structure.

    Parameters
    -----------
    :param filepath: the filepath to the map file.
    :type filepath: string

    :returns: xarray.DataArray
    """
    with xr.open_dataarray(f"{filepath}") as file:
        return file.load()


def _get_map_section(
    west_lon: float,
    east_lon: float,
    south_lat: float,
    north_lat: float,
    map_type: str = "relief",
    map_res: str = "02m",
    output_location: str = "./",
) -> None:
    """
    Function for querying the raw map source and saving it to a local file. This is the function
    that is called when running the module. The public method with the same name calls this function
    to act as the low-level interface.
    """
    west_lon = wrap_to_180(west_lon)
    east_lon = wrap_to_180(east_lon)
    # Validate map type and construct GMT map name to call via grdcut
    map_name = "earth_"
    if map_type == "gravity" and _validate_gravity_resolution(map_res):
        map_name += f"faa_{map_res}"
    elif map_type == "magnetic" and _validate_magentic_resolution(map_res):
        map_name += f"mag_{map_res}"
    elif map_type == "relief" and _validate_relief_resoltion(map_res):
        map_name += f"{map_type}_{map_res}"
    else:
        print("Map type not recognized")
        return

    if map_type == "relief" and (map_res == "03s" or map_res == "01s"):
        map_name += "g"
    else:
        map_name += "p"

    cmd = f"gmt grdcut @{map_name} -Rd{west_lon}/{east_lon}/{south_lat}/{north_lat} -G{output_location}.nc"
    out = subprocess.run(f"conda run -n PyGMT {cmd}", capture_output=True, shell=True)
    print(out.stdout.decode())
    return None


def main() -> None:
    parser = ArgumentParser(
        prog="GMT Map Access Tool",
        description="A light weight wrapper for accesssing GMT maps via Python.",
    )
    parser.add_argument(
        "--type",
        default="relief",
        choices=["relief", "gravity", "grav", "magnetic", "mag"],
        required=True,
        help="Map type to load.",
    )
    parser.add_argument(
        "--res",
        default="02m",
        required=False,
        help="Map resolution code. Available resolutions depend on the map selected.\nGravity:\t01d, 30m, 20m, 15m, 10m, 06m, 05m, 04m, 03m, 02m, 01m\nMagnetic:\t01d, 30m, 20m, 15m, 10m, 06m, 05m, 04m, 03m, 02m\nRelief:\t01d, 30m, 20m, 15m, 10m, 06m, 05m, 04m, 03m, 02m, 01m, 30s, 15s, 03s, 01s",
    )
    parser.add_argument(
        "--location", default="./", required=False, help="File location to save output."
    )
    parser.add_argument(
        "--name", default="map", required=False, help="Output file name."
    )

    args = parser.parse_args()
    _get_map_section()
    return None


def _validate_gravity_resolution(res: str) -> bool:
    valid = [
        "01d",
        "30m",
        "20m",
        "15m",
        "10m",
        "06m",
        "05m",
        "04m",
        "03m",
        "02m",
        "01m",
    ]
    if any(res == R for R in valid):
        return True
    else:
        print("Invalid resolution for map type: GRAVITY")
        return False


def _validate_magentic_resolution(res: str) -> bool:
    valid = ["01d", "30m", "20m", "15m", "10m", "06m", "05m", "04m", "03m", "02m"]
    if any(res == R for R in valid):
        return True
    else:
        print("Invalid resolution for map type: MAGNETIC")
        return False


def _validate_relief_resoltion(res: str) -> bool:
    valid = [
        "1d",
        "30m",
        "20m",
        "15m",
        "10m",
        "06m",
        "05m",
        "04m",
        "03m",
        "02m",
        "01m",
        "30s",
        "15s",
        "03s",
        "01s",
    ]
    if any(res == R for R in valid):
        return True
    else:
        print("Invalid resolution for map type: RELIEF")
        return False


if __name__ == "__main__":
    main()
