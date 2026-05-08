import kagglehub

# Download latest version
path = kagglehub.dataset_download("emmarex/plantdisease")

print("Path to dataset files:", path)

# copying the downloaded dataset to the project data folder
project_data_path = "data/"
shutil.copytree(downloaded_path, project_data_path, dirs_exist_ok=True)

print("Copied to:", project_data_path)