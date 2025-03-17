import os
import shutil

# Define the base path
base_path = r"C:\Users\apalaci6.URMC-SH\Box\my box\LALOR LAB\oscillations project\MATLAB\Warped Speech\stimuli"
# Define the redundant directory and the target directory
nums_to_remove=['0.000','3.940','6someshit']
warp_dirs=['median_rule5_seg_bark','lquartile_rule5_seg_bark','uquartile_rule5_seg_bark']
for ii,n_remove in enumerate(nums_to_remove):
    redundant_dir = os.path.join(base_path, f"stretchy_compressy\{warp_dirs[ii]}\wrinkle\stretchy_irreg\{n_remove}")

    target_dir = os.path.join(base_path, f"wrinkle\stretchy_compressy\stretchy_irreg\{warp_dirs[ii]}")
    os.makedirs(target_dir,exist_ok=True)
    # Move files from redundant directory to target directory
    for filename in os.listdir(redundant_dir):
        file_path = os.path.join(redundant_dir, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, target_dir)

    # Remove the redundant directory
    # os.rmdir(redundant_dir)

    # # Define the source and new paths for reorganization
    # current_path = os.path.join(base_path, "stretchy_compressy")
    # new_path = os.path.join(base_path, "wrinkle", "stretchy_compressy","stretchy_irreg")

    # # Create the new target directories if they don't exist
    # os.makedirs(new_path, exist_ok=True)

    # Move the directory
    # shutil.move(current_path, new_path)

print("Files moved and directories reorganized successfully!")
