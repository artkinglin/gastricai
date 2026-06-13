import os
import pydicom
import cv2
from tqdm import tqdm

input_folder = "data/raw_dicom"
output_folder = "data/processed_png"
os.makedirs(output_folder, exist_ok=True)

for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith(".dcm"):
        path = os.path.join(input_folder, filename)
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array

        # Normalize image to 0–255
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype('uint8')

        # Save as PNG
        out_path = os.path.join(output_folder, filename.replace(".dcm", ".png"))
        cv2.imwrite(out_path, img)

print("✅ Conversion complete! All DICOMs saved as PNGs in:", output_folder)