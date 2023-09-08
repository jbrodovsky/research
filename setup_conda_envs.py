"""
This is a Python script that will create the required conda environments that I typically use with my research. 
While I like Anaconda Navigator, it (depending on the configuration of your IT settings) can be buggy, and 
sometimes it is simply best to do a fresh re-install and re-build the environments from scratch. Given that, in
such a case, the base version of Conda itself may be out of date, we cannot rely on using the specific Python
API built into conda and should instead call it via the command line interface (which we will acces from Python
via the subprocess module). The base conda environment should be seperately and manually maintained.

The current list of base packages:

* numpy 
* scipy 
* matplotlib 
* pandas 
* scikit-learn 
* sympy 
* seaborn 
* jupyter 
* notebook 
* h5py 
* tqdm 
* pandoc 
* black 
* pytest 
* sphinx
"""

import subprocess, os


def conda_environment_exists(env_name: str) -> bool:
    """
    Check to see if the `conda` environment `env_name` exists.

    Parameters
    ----------
    :param env_name: the environment name to check.
    :type env_name: string

    :returns: True if the environment exists, false otherwise.
    """
    try:
        result = subprocess.run(
            f"conda info --envs", capture_output=True, text=True, check=True
        )
        output_lines = result.stdout.strip().split("\n")
        for line in output_lines:
            if line.strip().endswith(env_name):
                return True
        return False
    except subprocess.CalledProcessError:
        return False


def create_conda_environment(env_name: str, python_version, packages: str) -> None:
    """
    Create the `conda` environment `env_name` with a given version of Python and a set of additional packages to install.

    Parameters
    ----------
    :param env_name: the environment name to check.
    :type env_name: string
    :param python_version: the version number of Python to install.
    :type python_version: string
    :param packages: a space-separated string of packages to install
    :type pacakges: string

    :returns: None
    """
    if conda_environment_exists(env_name):
        print(f"<{env_name}> already exists")
        print("Checking to see if all desired packages are installed")
        packages_to_install = ""
        new_packages = False
        for pkg in f"{packages}".split():
            if not is_package_installed(env_name, pkg):
                print(f"<{pkg}> not found")
                packages_to_install += f"{pkg} "
                new_packages = True
        if new_packages:
            print(f"New packages to install: {packages_to_install}")
            subprocess.run(f"conda install {packages_to_install} -y")
            print("All packages installed successfully")
        else:
            print("No new packages needed.")
            val = input(f"Would you like to run an update on {env_name} y/n?")
            val = val.lower()
            if val == "y" or val == "yes":
                subprocess.run(f"conda update --all -y")
    else:
        print(f"Creating conda environment: <{env_name}>")
        try:
            subprocess.run(
                f"conda create -n {env_name} python={python_version} {packages} -y"
            )
            print(f"Environment <{env_name}> successfully created.")
        except:
            print(f"Error creating environment <{env_name}>")


def is_package_installed(env_name: str, package_name: str):
    """
    Check to see if the given package `package_name` is installed in the `conda` environment `evn_name`.

    Parameters
    ----------
    :param env_name: the environment name to check.
    :type env_name: string
    :param package_name: the package name to check.
    :type package_name: string
    """
    try:
        result = subprocess.run(
            f"conda list -n {env_name} {package_name}",
            capture_output=True,
            text=True,
            check=True,
        )
        output_lines = result.stdout.strip().split("\n")
        for line in output_lines:
            if line.strip().startswith(package_name):
                return True
        return False
    except subprocess.CalledProcessError:
        return False


def main():
    """
    Runs this module as a script and checks/creates the default specified environments. Might eventually switch this to reading in some sort of json or plain text file.
    """
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser(
        prog="Conda Environment Setup Utility",
        description="Sets up and pseudo-maintains a set of conda environments from a json file.",
    )
    parser.add_argument(
        "--config", help="Path to environment configuration json file", required=True
    )
    args = parser.parse_args()

    try:
        # Open the JSON file in read mode
        with open(args.config, "r") as file:
            # Load the JSON data into a Python data structure
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{args.config}' does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error: JSON decoding failed. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    else:
        # The code inside the 'else' block will only execute if no exceptions were raised.
        print("JSON data loaded successfully.")
        print(data)
    environments = data["environments"]
    base_packages = data["base_packages"]

    for env_name in environments.keys():
        create_conda_environment(
            env_name,
            environments[env_name]["python"],
            f"{base_packages} {environments[env_name]['additional_packages']}",
        )
    print("Setup complete!")


if __name__ == "__main__":
    main()
