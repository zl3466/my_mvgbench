import math
import json
import numpy as np
import sys
import os
import shutil
from glob import glob
from tqdm import tqdm



def look_at(eye, target, up=(0, 1, 0)):
    """
    eye: camera position
    target: target position
    returns a world to camera transform matrix 4x4
    """
    # Convert to numpy arrays
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)

    # Compute forward, right and up vectors
    forward = eye - target # forward: target --> camera, opposite to opengl camera
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    # Create rotation and translation matrices
    rotation = np.eye(4)
    rotation[:3, :3] = np.vstack([right, up, forward]) # vstack: forward is the 3rd row

    translation = np.eye(4)
    translation[:3, 3] = -eye

    # Combine rotation and translation to form view matrix
    view_matrix = np.dot(rotation, translation)

    return view_matrix

# Load JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data

def get_camera_to_world_transform(lookat_pos, orbital_radius, angle_deg, camera_height_offset, up=(0, 0, 1)):
    """
    Compute 4x4 camera-to-world transformation matrix.
    
    Args:
        lookat_pos: [x, y, z] position the camera looks at
        orbital_radius: distance from lookat_pos to camera
        angle_deg: orbital angle in degrees (azimuth)
        camera_height_offset: height offset for camera position
        up: up vector (default (0, 0, 1) for Z-up coordinate system)
    
    Returns:
        4x4 numpy array representing camera-to-world transformation matrix
    """
    # Convert angle to radians
    angle_rad = math.radians(angle_deg)
    
    # Calculate camera position in orbital pattern
    cam_x = lookat_pos[0] + orbital_radius * math.cos(angle_rad)
    cam_y = lookat_pos[1] + orbital_radius * math.sin(angle_rad)
    cam_z = lookat_pos[2] + camera_height_offset
    cam_pos = np.array([cam_x, cam_y, cam_z])
    
    # look_at returns world-to-camera (w2c) matrix
    # We need camera-to-world (c2w), so we invert it
    w2c = look_at(cam_pos, lookat_pos, up=up)
    c2w = np.linalg.inv(w2c)
    
    return c2w

# Load and concatenate all JSONL files, then sort by sample_id
def load_jsonl(file_path):
    """Load a JSONL file and return a list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def get_file_path(filename):
    """Convert filename to 3dgs file_path format: remove extension and add ./ prefix"""
    # Remove extension (e.g., 'image.png' -> 'image')
    name_without_ext = os.path.splitext(filename)[0]
    # Add ./ prefix as shown in the example
    return f"./{name_without_ext}"


camera_height_offset = 0.25
orbital_radius = 0.75

orbital_angles_degrees = [45, 135, 225, 315]   
orbital_angles_degrees_8 = [0, 45, 90, 135, 180, 225, 270, 315]

FOV_DEGREES = 30
camera_angle_x = math.radians(FOV_DEGREES)  # Convert horizontal FOV to radians

run_name = "gpt5"
input_dir = f"/Users/zhihengli/Downloads/mvgbench_dataset/gpt5"
output_dir = f"/Users/zhihengli/Downloads/mvgbench_dataset/run_gpt5"

view_idx_list = [1, 3, 5, 7]

jsonl_file = f"{input_dir}/view_1/results_merged.jsonl"
entries = load_jsonl(jsonl_file)

print(f"Loaded {len(entries)} entries")

# for each object sample
# for sample_record in tqdm(all_entries[:4], desc="Processing samples"):
for j in tqdm(range(len(entries)), desc="Processing samples"):
    sample_id = j + 1
    sample_record = entries[j]
    lookat_pos = np.array(sample_record['lookat_pos'])

    # for each view angle
    final_state_img_list_even = []
    final_state_img_list_odd = []
    middle_state_img_list_even = []
    middle_state_img_list_odd = []

    trans_even = []
    trans_odd = []

    for i in range(len(view_idx_list)):
        view_idx = view_idx_list[i]
        angle_deg = orbital_angles_degrees[i]

        c2w = get_camera_to_world_transform(
            lookat_pos, 
            orbital_radius, 
            angle_deg, 
            camera_height_offset,
            up=(0, 0, 1)  # Z-up coordinate system
        )
        # transforms.append(c2w)
        img_folder = f"{input_dir}/view_{view_idx}/{sample_id}"

        img_list = sorted(
            [fname for fname in os.listdir(img_folder) if fname.startswith('output_image_') and fname.lower().endswith('.png')],
            key=lambda f: int(''.join(filter(str.isdigit, f)))
        )
        final_state_img = img_list[-1] if len(img_list) < 10 else img_list[9]
        middle_state_img = img_list[len(img_list) // 2]

        if i % 2 == 0:
            final_state_img_list_even.append(f"{img_folder}/{final_state_img}")
            middle_state_img_list_even.append(f"{img_folder}/{middle_state_img}")
            trans_even.append(c2w)
        else:
            final_state_img_list_odd.append(f"{img_folder}/{final_state_img}")
            middle_state_img_list_odd.append(f"{img_folder}/{middle_state_img}")
            trans_odd.append(c2w)
        # final_state_img_list.append(f"{img_folder}/{final_state_img}")
        # middle_state_img_list.append(f"{img_folder}/{middle_state_img}")

    # copy images into output_dir
    final_state_even_folder = f"{output_dir}/{run_name}_final_even/{sample_id}"
    final_state_odd_folder = f"{output_dir}/{run_name}_final_odd/{sample_id}"
    middle_state_even_folder = f"{output_dir}/{run_name}_middle_even/{sample_id}"
    middle_state_odd_folder = f"{output_dir}/{run_name}_middle_odd/{sample_id}"
    os.makedirs(final_state_even_folder, exist_ok=True)
    os.makedirs(final_state_odd_folder, exist_ok=True)
    os.makedirs(middle_state_even_folder, exist_ok=True)
    os.makedirs(middle_state_odd_folder, exist_ok=True)
    
    # Track new filenames for transforms_train.json
    final_state_even_train_filenames = []
    final_state_odd_train_filenames = []
    middle_state_even_train_filenames = []
    middle_state_odd_train_filenames = []

    final_state_even_test_filenames = []
    final_state_odd_test_filenames = []
    middle_state_even_test_filenames = []
    middle_state_odd_test_filenames = []
    
    # Copy final_state images (even)
    for idx, img_path in enumerate(final_state_img_list_even):
        # copy train image with idx-based filename
        img_name = os.path.basename(img_path)
        img_ext = os.path.splitext(img_name)[1]
        new_train_name = f"{idx:03d}{img_ext}"
        dst_path = os.path.join(final_state_even_folder, new_train_name)
        shutil.copy2(img_path, dst_path)
        final_state_even_train_filenames.append(new_train_name)
        # copy test images
        if idx == 0:
            first_img_name_without_ext = os.path.splitext(img_name)[0]
            first_img_ext = os.path.splitext(img_name)[1]
            for i in range(len(view_idx_list)):
                test_name = f"test_{i:03d}{first_img_ext}"
                dst_path = os.path.join(final_state_even_folder, test_name)
                shutil.copy2(img_path, dst_path)
                final_state_even_test_filenames.append(test_name)
    
    # Copy final_state images (odd)
    for idx, img_path in enumerate(final_state_img_list_odd):
        # copy train image with idx-based filename
        img_name = os.path.basename(img_path)
        img_ext = os.path.splitext(img_name)[1]
        new_train_name = f"{idx:03d}{img_ext}"
        dst_path = os.path.join(final_state_odd_folder, new_train_name)
        shutil.copy2(img_path, dst_path)
        final_state_odd_train_filenames.append(new_train_name)
        # copy test images
        if idx == 0:
            first_img_name_without_ext = os.path.splitext(img_name)[0]
            first_img_ext = os.path.splitext(img_name)[1]
            for i in range(len(view_idx_list)):
                test_name = f"test_{i:03d}{first_img_ext}"
                dst_path = os.path.join(final_state_odd_folder, test_name)
                shutil.copy2(img_path, dst_path)
                final_state_odd_test_filenames.append(test_name)
    
    # Copy middle_state images (even)
    for idx, img_path in enumerate(middle_state_img_list_even):
        # copy train image with idx-based filename
        img_name = os.path.basename(img_path)
        img_ext = os.path.splitext(img_name)[1]
        new_train_name = f"{idx:03d}{img_ext}"
        dst_path = os.path.join(middle_state_even_folder, new_train_name)
        shutil.copy2(img_path, dst_path)
        middle_state_even_train_filenames.append(new_train_name)
        # copy test images
        if idx == 0:
            first_img_name_without_ext = os.path.splitext(img_name)[0]
            first_img_ext = os.path.splitext(img_name)[1]
            for i in range(len(view_idx_list)):
                test_name = f"test_{i:03d}{first_img_ext}"
                dst_path = os.path.join(middle_state_even_folder, test_name)
                shutil.copy2(img_path, dst_path)
                middle_state_even_test_filenames.append(test_name)
    
    # Copy middle_state images (odd)
    for idx, img_path in enumerate(middle_state_img_list_odd):
        # copy train image with idx-based filename
        img_name = os.path.basename(img_path)
        img_ext = os.path.splitext(img_name)[1]
        new_train_name = f"{idx:03d}{img_ext}"
        dst_path = os.path.join(middle_state_odd_folder, new_train_name)
        shutil.copy2(img_path, dst_path)
        middle_state_odd_train_filenames.append(new_train_name)
        # copy test images
        if idx == 0:
            first_img_name_without_ext = os.path.splitext(img_name)[0]
            first_img_ext = os.path.splitext(img_name)[1]
            for i in range(len(view_idx_list)):
                test_name = f"test_{i:03d}{first_img_ext}"
                dst_path = os.path.join(middle_state_odd_folder, test_name)
                shutil.copy2(img_path, dst_path)
                middle_state_odd_test_filenames.append(test_name)
    
    # save transforms_train.json for even and odd
    # transforms are the same for final and middle state images
    # trans_even = [transforms[i] for i in range(0, len(transforms), 2)]
    # trans_odd = [transforms[i] for i in range(1, len(transforms), 2)]
    transforms_train_even_final = {
        "camera_angle_x": camera_angle_x,  # REQUIRED - horizontal FOV in radians
        "frames": [
            {
                "file_path": get_file_path(final_state_even_train_filenames[i]),
                "transform_matrix": trans_even[i].tolist()
            }
            for i in range(len(trans_even))
        ]
    }
    with open(f"{final_state_even_folder}/transforms_train.json", "w") as f:
        json.dump(transforms_train_even_final, f, indent=2)
    
    transforms_train_even_middle = {
        "camera_angle_x": camera_angle_x,  # REQUIRED - horizontal FOV in radians
        "frames": [
            {
                "file_path": get_file_path(middle_state_even_train_filenames[i]),
                "transform_matrix": trans_even[i].tolist()
            }
            for i in range(len(trans_even))
        ]
    }
    with open(f"{middle_state_even_folder}/transforms_train.json", "w") as f:
        json.dump(transforms_train_even_middle, f, indent=2)

    transforms_train_odd_final = {
        "camera_angle_x": camera_angle_x,  # REQUIRED - horizontal FOV in radians
        "frames": [
            {
                "file_path": get_file_path(final_state_odd_train_filenames[i]),
                "transform_matrix": trans_odd[i].tolist()
            }
            for i in range(len(trans_odd))
        ]
    }
    with open(f"{final_state_odd_folder}/transforms_train.json", "w") as f:
        json.dump(transforms_train_odd_final, f, indent=2)
    
    transforms_train_odd_middle = {
        "camera_angle_x": camera_angle_x,  # REQUIRED - horizontal FOV in radians
        "frames": [
            {
                "file_path": get_file_path(middle_state_odd_train_filenames[i]),
                "transform_matrix": trans_odd[i].tolist()
            }
            for i in range(len(trans_odd))
        ]
    }
    with open(f"{middle_state_odd_folder}/transforms_train.json", "w") as f:
        json.dump(transforms_train_odd_middle, f, indent=2)

    # Create transforms_test.json for even and odd folders with test images
    # Use the collected test filenames
    transforms_test_even_final = {
        "camera_angle_x": camera_angle_x,
        "frames": [
            {
                "file_path": get_file_path(final_state_even_test_filenames[i]),
                "transform_matrix": trans_even[0].tolist() # test images are the first image repeated n times
            }
            for i in range(len(final_state_even_test_filenames))
        ]
    }
    
    transforms_test_even_middle = {
        "camera_angle_x": camera_angle_x,
        "frames": [
            {
                "file_path": get_file_path(middle_state_even_test_filenames[i]),
                "transform_matrix": trans_even[0].tolist() # test images are the first image repeated n times
            }
            for i in range(len(middle_state_even_test_filenames))
        ]
    }
    
    transforms_test_odd_final = {
        "camera_angle_x": camera_angle_x,
        "frames": [
            {
                "file_path": get_file_path(final_state_odd_test_filenames[i]),
                "transform_matrix": trans_odd[0].tolist() # test images are the first image repeated n times
            }
            for i in range(len(final_state_odd_test_filenames))
        ]
    }
    
    transforms_test_odd_middle = {
        "camera_angle_x": camera_angle_x,
        "frames": [
            {
                "file_path": get_file_path(middle_state_odd_test_filenames[i]),
                "transform_matrix": trans_odd[0].tolist() # test images are the first image repeated n times
            }
            for i in range(len(middle_state_odd_test_filenames))
        ]
    }

    with open(f"{final_state_even_folder}/transforms_test.json", "w") as f:
        json.dump(transforms_test_even_final, f, indent=2)
    with open(f"{middle_state_even_folder}/transforms_test.json", "w") as f:
        json.dump(transforms_test_even_middle, f, indent=2)
    
    with open(f"{final_state_odd_folder}/transforms_test.json", "w") as f:
        json.dump(transforms_test_odd_final, f, indent=2)
    with open(f"{middle_state_odd_folder}/transforms_test.json", "w") as f:
        json.dump(transforms_test_odd_middle, f, indent=2)
    
    # Copy transforms_test.json to transforms_val.json in both folders
    shutil.copy2(f"{final_state_even_folder}/transforms_test.json", f"{final_state_even_folder}/transforms_val.json")
    shutil.copy2(f"{final_state_odd_folder}/transforms_test.json", f"{final_state_odd_folder}/transforms_val.json")
    shutil.copy2(f"{middle_state_even_folder}/transforms_test.json", f"{middle_state_even_folder}/transforms_val.json")
    shutil.copy2(f"{middle_state_odd_folder}/transforms_test.json", f"{middle_state_odd_folder}/transforms_val.json")




# img_list = sorted(
#     [fname for fname in os.listdir(input_img_folder) if fname.lower().endswith('.png')],
#     key=lambda f: int(''.join(filter(str.isdigit, f)))
# )

# # ========== create transformation json ==========
# # Load all entries from the JSONL file
# json_data = json.load(open(input_json_path))
# lookat_pos = np.array(json_data['lookat_pos'])

# # Generate transformation matrices
# if len(img_list) == 8:
#     orbital_angles_degrees = orbital_angles_degrees_8
# elif len(img_list) == 4:
#     orbital_angles_degrees = orbital_angles_degrees
# else:
#     raise ValueError(f"Unsupported number of images: {len(img_list)}")

# transforms = []
# for angle_idx, angle_deg in enumerate(orbital_angles_degrees):
#     c2w = get_camera_to_world_transform(
#         lookat_pos, 
#         orbital_radius, 
#         angle_deg, 
#         camera_height_offset,
#         up=(0, 0, 1)  # Z-up coordinate system
#     )
#     transforms.append(c2w)
#     print(f"{angle_deg}°: {c2w}")

# # Split into even and odd indices
# img_list_even = [img_list[i] for i in range(0, len(img_list), 2)]  # indices 0, 2, 4, ...
# img_list_odd = [img_list[i] for i in range(1, len(img_list), 2)]   # indices 1, 3, 5, ...

# trans_even = [transforms[i] for i in range(0, len(transforms), 2)]
# trans_odd = [transforms[i] for i in range(1, len(transforms), 2)]

# print(f"len(img_list_even): {len(img_list_even)}")
# print(f"len(img_list_odd): {len(img_list_odd)}")
# print(f"len(trans_even): {len(trans_even)}")
# print(f"len(trans_odd): {len(trans_odd)}")

# # Create even and odd folders
# even_folder = os.path.join(input_dir, f'eval_ready_{run_name}', f'{run_name}_even', sample_id)
# odd_folder = os.path.join(input_dir, f'eval_ready_{run_name}', f'{run_name}_odd', sample_id)
# os.makedirs(even_folder, exist_ok=True)
# os.makedirs(odd_folder, exist_ok=True)

# # Copy images to respective folders
# for img_name in img_list_even:
#     src_path = os.path.join(input_img_folder, img_name)
#     dst_path = os.path.join(even_folder, img_name)
#     shutil.copy2(src_path, dst_path)

# for img_name in img_list_odd:
#     src_path = os.path.join(input_img_folder, img_name)
#     dst_path = os.path.join(odd_folder, img_name)
#     shutil.copy2(src_path, dst_path)

# # Copy first image len(img_list) times as test images
# if len(img_list) > 0:
#     first_img = img_list[0]
#     first_img_path = os.path.join(input_img_folder, first_img)
#     first_img_name_without_ext = os.path.splitext(first_img)[0]
#     first_img_ext = os.path.splitext(first_img)[1]
    
#     # Copy to even folder
#     for i in range(len(img_list)):
#         test_name = f"test_{first_img_name_without_ext}_{i:03d}{first_img_ext}"
#         dst_path = os.path.join(even_folder, test_name)
#         shutil.copy2(first_img_path, dst_path)
    
#     # Copy to odd folder
#     for i in range(len(img_list)):
#         test_name = f"test_{first_img_name_without_ext}_{i:03d}{first_img_ext}"
#         dst_path = os.path.join(odd_folder, test_name)
#         shutil.copy2(first_img_path, dst_path)
    
#     print(f"Copied first image '{first_img}' {len(img_list)} times to both even and odd folders as test images")

# # save transforms_train.json for even and odd
# # NOTE: camera_angle_x is REQUIRED by the 3DGS pipeline (in radians)
# # This is the horizontal field of view. Adjust based on your camera setup.


# def get_file_path(filename):
#     """Convert filename to file_path format: remove extension and add ./ prefix"""
#     # Remove extension (e.g., 'image.png' -> 'image')
#     name_without_ext = os.path.splitext(filename)[0]
#     # Add ./ prefix as shown in the example
#     return f"./{name_without_ext}"

# transforms_train_even = {
#     "camera_angle_x": camera_angle_x,  # REQUIRED - horizontal FOV in radians
#     "frames": [
#         {
#             "file_path": get_file_path(img_list_even[i]),
#             "transform_matrix": trans_even[i].tolist()
#         }
#         for i in range(len(trans_even))
#     ]
# }

# with open(f"{even_folder}/transforms_train.json", "w") as f:
#     json.dump(transforms_train_even, f, indent=2)

# transforms_train_odd = {
#     "camera_angle_x": camera_angle_x,  # REQUIRED - horizontal FOV in radians
#     "frames": [
#         {
#             "file_path": get_file_path(img_list_odd[i]),
#             "transform_matrix": trans_odd[i].tolist()
#         }
#         for i in range(len(trans_odd))
#     ]
# }

# with open(f"{odd_folder}/transforms_train.json", "w") as f:
#     json.dump(transforms_train_odd, f, indent=2)

# # Create transforms_test.json for even and odd folders with test images
# if len(img_list) > 0:
#     first_img_name_without_ext = os.path.splitext(img_list[0])[0]
#     first_img_ext = os.path.splitext(img_list[0])[1]
    
#     # For even folder: use all transforms (since we have len(img_list) test images)
#     transforms_test_even = {
#         "camera_angle_x": camera_angle_x,
#         "frames": [
#             {
#                 "file_path": get_file_path(f"test_{first_img_name_without_ext}_{i:03d}{first_img_ext}"),
#                 "transform_matrix": transforms[i].tolist()
#             }
#             for i in range(len(transforms))
#         ]
#     }
    
#     # For odd folder: use all transforms (since we have len(img_list) test images)
#     transforms_test_odd = {
#         "camera_angle_x": camera_angle_x,
#         "frames": [
#             {
#                 "file_path": get_file_path(f"test_{first_img_name_without_ext}_{i:03d}{first_img_ext}"),
#                 "transform_matrix": transforms[i].tolist()
#             }
#             for i in range(len(transforms))
#         ]
#     }
    
#     with open(f"{even_folder}/transforms_test.json", "w") as f:
#         json.dump(transforms_test_even, f, indent=2)
    
#     with open(f"{odd_folder}/transforms_test.json", "w") as f:
#         json.dump(transforms_test_odd, f, indent=2)
    
#     # Copy transforms_test.json to transforms_val.json in both folders
#     shutil.copy2(f"{even_folder}/transforms_test.json", f"{even_folder}/transforms_val.json")
#     shutil.copy2(f"{odd_folder}/transforms_test.json", f"{odd_folder}/transforms_val.json")
    
#     print(f"Created transforms_test.json for both even and odd folders with {len(transforms)} test image entries")
#     print(f"Copied transforms_test.json to transforms_val.json in both folders")

# print(f"Copied {len(img_list_even)} images to {even_folder}")
# print(f"Copied {len(img_list_odd)} images to {odd_folder}")



    


