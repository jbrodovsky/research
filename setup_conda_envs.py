"""
This is a Python scripte that will create the required conda environments that I typically use with my research. 
While I like Anaconda Navigator, it (depending on the configuration of your IT settings) can be buggy, and 
sometimes it is simply best to do a fresh re-install and re-build the environments from scratch. Given that, in
such a case, the base version of Conda itself may be out of date, we cannot rely on using the specific Python
API built into conda and should instead call it via the command line interface (which we will acces from Python
via the subprocess module). The base conda environment should be seperately and manually maintained.
"""

# List of base packages I like to use withing every environment
base_packages = "numpy scipy matplotlib pandas scikit-learn sympy seaborn jupyter notebook h5py tqdm pandoc black pytest sphinx"

# List of environments to create along with their additional
environments = {
    "python311": {"python": 3.11, "additional_packages": ""},
    "geophysical_nav": {
        "python": 3.11,
        "additional_packages": "haversine xarray cython",
    },
    "PyGMT": {"python": "3.8", "additional_packages": "pygmt"},
}

# ----------------------------------------------------------------
import subprocess, os


def conda_environment_exists(env_name: str):
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


def create_conda_environment(env_name: str, python_version, packages: str):
    if conda_environment_exists(env_name):
        print(f"<{env_name}> already exists")
        print("Checking to see if all desired packages are installed")
        packages_to_install = ""
        new_packages = False
        for pkg in f"{base_packages} {packages}".split():
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
                f"conda create -n {env_name} python={python_version} {base_packages} {packages} -y"
            )
            print(f"Environment <{env_name}> successfully created.")
        except:
            print(f"Error creating environment <{env_name}>")


def is_package_installed(env_name: str, package_name: str):
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
    for env_name in environments.keys():
        create_conda_environment(
            env_name,
            environments[env_name]["python"],
            environments[env_name]["additional_packages"],
        )
    print("Setup complete!")


if __name__ == "__main__":
    main()
