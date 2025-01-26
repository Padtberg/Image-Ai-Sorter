import os
import shutil
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Categories for classification and keywords
CATEGORIES = ["Beach", "Sea", "Ocean", "Sunset", "Mountains", "City", "Forest", "Animals", "People", "Sky", "Landscape"]

def extract_keywords(image_path):
    """Extract descriptive keywords for an image."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=CATEGORIES, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).tolist()[0]
        
        # Get top 4 keywords based on probabilities
        top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:4]
        keywords = [CATEGORIES[i] for i in top_indices]
        return keywords
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def rename_file_with_keywords(dest_path, keywords):
    """Rename a file with descriptive keywords."""
    try:
        # Join the keywords with commas
        description = ", ".join(keywords)
        
        # Generate a new name
        folder, _ = os.path.split(dest_path)
        new_name = f"{description}.jpg"
        new_path = os.path.join(folder, new_name)
        
        # Rename the file
        os.rename(dest_path, new_path)
        print(f"Renamed file to: {new_name}")
    except Exception as e:
        print(f"Error renaming {dest_path}: {e}")

def organize_and_rename_images(sd_card_path, hard_disk_path):
    """Organize images into folders, rename them with descriptive keywords, and handle unclassified images."""
    if not os.path.exists(sd_card_path):
        print("Source path does not exist.")
        return

    # Ensure destination path exists
    Path(hard_disk_path).mkdir(parents=True, exist_ok=True)

    # Path for unclassified images
    unclassified_path = os.path.join(hard_disk_path, "Unclassified")
    Path(unclassified_path).mkdir(parents=True, exist_ok=True)

    # Loop through all image files on the SD card
    for root, _, files in os.walk(sd_card_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                src_path = os.path.join(root, file)

                # Extract keywords for the image
                keywords = extract_keywords(src_path)

                if keywords:
                    # Use the first keyword as the main category
                    main_category = keywords[0]

                    # Create the category folder if it doesn't exist
                    category_path = os.path.join(hard_disk_path, main_category)
                    Path(category_path).mkdir(parents=True, exist_ok=True)

                    # Copy the file to the appropriate folder
                    dest_path = os.path.join(category_path, file)
                    shutil.copy2(src_path, dest_path)
                    print(f"Moved {file} to {main_category} folder.")

                    # Rename the file with descriptive keywords
                    rename_file_with_keywords(dest_path, keywords)
                else:
                    # Move the unclassified file to the "Unclassified" folder
                    dest_path = os.path.join(unclassified_path, file)
                    shutil.copy2(src_path, dest_path)
                    print(f"Moved {file} to Unclassified folder.")

def select_folder(prompt):
    """Open a file dialog to select a folder."""
    Tk().withdraw()  # Hide the root Tkinter window
    folder = askdirectory(title=prompt)
    if not folder:
        print("No folder selected. Exiting.")
        exit()
    return folder

if __name__ == "__main__":
    print("Please select the source folder (e.g., your SD card).")
    SD_CARD_PATH = select_folder("Select the source folder (SD card)")

    print("Please select the destination folder (e.g., where sorted folders will be created).")
    HARD_DISK_PATH = select_folder("Select the destination folder")

    print(f"Source folder: {SD_CARD_PATH}")
    print(f"Destination folder: {HARD_DISK_PATH}")

    organize_and_rename_images(SD_CARD_PATH, HARD_DISK_PATH)
