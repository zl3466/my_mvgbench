import os
from PIL import Image
from glob import glob
from tqdm import tqdm

input_dir = "/Users/zhihengli/Downloads/mvgbench_dataset/v7_mini/output_images_dummy"
output_dir = "/Users/zhihengli/Downloads/mvgbench_dataset/v7_mini/output_images"
sample_list = os.listdir(input_dir)

def flip_and_save_images(input_folder, output_folder):
    """
    Horizontally flip (mirror) all images from input folder and save to output folder.
    
    Args:
        input_folder: Path to the input folder containing images
        output_folder: Path to the output folder where flipped images will be saved
    
    Returns:
        Number of images flipped and saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    
    # Find all image files in the input folder
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(input_folder, ext)))
    
    if not image_files:
        return 0
    
    # Flip each image and save to output folder
    flipped_count = 0
    for img_path in image_files:
        try:
            # Open the image
            img = Image.open(img_path)
            
            # Horizontally flip (mirror) the image
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Get the filename and save to output folder
            img_filename = os.path.basename(img_path)
            output_path = os.path.join(output_folder, img_filename)
            flipped_img.save(output_path)
            flipped_count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return flipped_count

# Flip images from input folder and save to output folder
print(f"Reading images from: {input_dir}")
print(f"Saving flipped images to: {output_dir}")
print(f"Found {len(sample_list)} sample folders\n")

total_flipped = 0
for sample_folder in tqdm(sample_list, desc="Processing sample folders"):
    input_sample_path = os.path.join(input_dir, sample_folder)
    output_sample_path = os.path.join(output_dir, sample_folder)
    
    if os.path.isdir(input_sample_path):
        flipped_count = flip_and_save_images(input_sample_path, output_sample_path)
        total_flipped += flipped_count
        if flipped_count > 0:
            print(f"  {sample_folder}: flipped and saved {flipped_count} images")

print(f"\nTotal images flipped and saved: {total_flipped}")