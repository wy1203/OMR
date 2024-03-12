import tarfile
import os

# Paths to your archives
archive_path_1 = 'grandstaff-lmx.2024-02-12.tar.gz'
archive_path_2 = 'grandstaff.tgz'

# Base target directory for extraction
target_dir_base = 'dataset'

# Subdirectories for each archive's contents
subdir_1 = 'grandstaff_lmx'
subdir_2 = 'grandstaff_img'

# Ensure the target base directory exists
os.makedirs(target_dir_base, exist_ok=True)

# Ensure subdirectories exist
target_dir_1 = os.path.join(target_dir_base, subdir_1)
target_dir_2 = os.path.join(target_dir_base, subdir_2)
os.makedirs(target_dir_1, exist_ok=True)
os.makedirs(target_dir_2, exist_ok=True)

# Open and extract the first tar.gz archive to its specific subdirectory
with tarfile.open(archive_path_1, 'r:gz') as archive:
    archive.extractall(path=target_dir_1)
print(f'Files from {archive_path_1} extracted to {target_dir_1}')

# Open and extract the second tgz archive to its specific subdirectory
with tarfile.open(archive_path_2, 'r:gz') as archive:
    archive.extractall(path=target_dir_2)
print(f'Files from {archive_path_2} extracted to {target_dir_2}')