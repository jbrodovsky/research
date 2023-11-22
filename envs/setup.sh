#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    # Install conda
    echo "conda not found. Please install conda or manually create environments."
fi

# Create or update conda environments
for env_file in ./*.yml
do
    env_name=$(basename "$env_file" .yml)
    if conda env list | grep -q "$env_name"
    then
        echo "Updating environment $env_name..."
        conda env update --file "$env_file"
    else
        echo "Creating environment $env_name..."
        conda env create --file "$env_file"
    fi
done
