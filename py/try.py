import os
from PIL import Image

def pad_images_in_directory(input_dir, output_dir):
    # checks directory
    os.makedirs(output_dir, exist_ok=True)

    # loads images and gets max file size
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sizes = []

    for file in image_files:
        with Image.open(os.path.join(input_dir, file)) as img:
            sizes.append(img.size)  # (width, height)

    # get largest width and height
    max_width = max(w for w, h in sizes)
    max_height = max(h for w, h in sizes)

    print(f"Target size: {max_width}x{max_height}")

    # makes everything same size
    for file in image_files:
        with Image.open(os.path.join(input_dir, file)) as img:
            width, height = img.size

            # calc padding
            left = (max_width - width) // 2
            right = max_width - width - left
            top = (max_height - height) // 2
            bottom = max_height - height - top

            # makes new image with white 
            result = Image.new(img.mode, (max_width, max_height), (255, 255, 255))
            result.paste(img, (left, top))

            # saving
            save_path = os.path.join(output_dir, file)
            result.save(save_path)

            print(f"Padded and saved: {save_path}")


input_directory = "Data/Raw"
output_directory = "Data/Padded_Raw"
pad_images_in_directory(input_directory, output_directory)
