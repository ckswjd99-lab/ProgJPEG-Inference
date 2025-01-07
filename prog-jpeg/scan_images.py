from PIL import Image, ImageFile
import subprocess
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

def save_scan_stages(file_path, scans_path, output_folder):
    with open(scans_path, "r") as f:
        scans = [line for line in f if line.strip() and not line.strip().startswith("#")]

    scan_file_template = os.path.join(output_folder, "scan_stage_{stage}.jpg")
    for i, scan in enumerate(scans, start=1):
        temp_scans_path = os.path.join(output_folder, f"temp_scans_{i}.txt")
        with open(temp_scans_path, "w") as temp_scan_file:
            temp_scan_file.write("".join(scans[:i]))

        stage_output_path = scan_file_template.format(stage=i)
        jpegtran_command = f"jpegtran -scans {temp_scans_path} {file_path} > {stage_output_path}"
        subprocess.run(jpegtran_command, shell=True)
        print(f"Saved scan stage {i} to: {stage_output_path}")
    
        os.remove(temp_scans_path)

original_jpeg_path = "original.jpg"
custom_scans_path = "scan.txt"
output_folder = "output_images"

# 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

save_scan_stages(original_jpeg_path, custom_scans_path, output_folder)

print("All scan stage images saved in the output_images folder.")
