# Check if conda is installed
if (!(Get-Command conda -ErrorAction SilentlyContinue)) {
    # Install conda
    Write-Output "conda not found. Please install conda or manually create environments."
}

# Create or update conda environments
$env_files = Get-ChildItem -Path "." -Filter "*.yml"

foreach ($env_file in $env_files) {
    $env_name = [System.IO.Path]::GetFileNameWithoutExtension($env_file)
    if ((conda env list) -match $env_name) {
        Write-Output "Updating environment $env_name..."
        conda env update --file $env_file
    } else {
        Write-Output "Creating environment $env_name..."
        conda env create --file $env_file
    }
}