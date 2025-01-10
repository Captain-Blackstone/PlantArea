import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.cluster import KMeans

# Define the green color range in HSV
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

def calculate_green_percentage(image_path):
    """
    Calculate green percentage for an individual image.
    Returns a list of results for each box in the grid.
    """
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return []

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to remove noise
    smoothed = cv2.GaussianBlur(gray, (51, 51), 0)


    # Binarize to separate boxes from background
    _, binary = cv2.threshold(smoothed, 60, 255, cv2.THRESH_BINARY_INV)
    binary = 255 - binary

    # Find contours of boxes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(contour) for contour in contours]).reshape(-1, 1)
    if (areas.std()/areas.mean()) < 0.1:
        valid_contours = contours
    else:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(areas)
        centers = kmeans.cluster_centers_
        valid_contours = []
        for pos in np.where(labels == np.argmax(centers))[0]:
            valid_contours.append(contours[pos])
    grid_rectangles = []
    for contour in valid_contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) > 0:
            x, y, w, h = cv2.boundingRect(approx)
            grid_rectangles.append((x, y, w, h))
    grid_rectangles = sorted(grid_rectangles, key=lambda r: (r[1], r[0]))

    # Calculate green percentage for each box
    percentages = []
    for rect in grid_rectangles:
        x, y, w, h = rect
        box = image[y:y+h, x:x+w]
        hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        total_pixels = w * h
        green_pixels = cv2.countNonZero(green_mask)
        green_percentage = (green_pixels / total_pixels) * 100
        percentages.append((x, y, green_percentage))

    # Determine grid rows and columns
    results = []
    current_row = 0
    current_column = 0
    threshold = max(rect[3] for rect in grid_rectangles) * 0.5 if grid_rectangles else 0
    prev_y = None

    for i, (x, y, percentage) in enumerate(percentages):
        if prev_y is not None and abs(y - prev_y) >= threshold:
            current_row += 1
            current_column = 0
        elif prev_y is not None:
            current_column += 1
        prev_y = y
        results.append([Path(*image_path.parts[-2:]), current_row, current_column, percentage, f"{i+1}/{len(percentages)}"])
    return results

def process_images_in_folder(folder_path, output_csv):
    """
    Process all images in a folder and save results into a single CSV file.
    """
    folder_path = Path(folder_path)
    output_csv = Path(output_csv)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"The specified folder does not exist: {folder_path}")

    # Initialize results list
    all_results = []

    # Iterate over all image files in the folder
    for image_path in folder_path.glob("*"):
        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            print(f"Processing image: {image_path}...", end=" ")
            results = calculate_green_percentage(image_path)
            all_results.extend(results)
            print(f"detected {len(results)} boxes.")
        else:
            print(f"Skipping image: {image_path} (unsupported file extension)")
    
    # Save all results to a single CSV file
    if all_results:
        columns = ["filename", "row", "column", "percentage", "box#"]
        df = pd.DataFrame(all_results, columns=columns)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    else:
        print("No valid images were found in the folder.")

if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(description="Calculate green area percentage in grid boxes for all images in a folder.")
    parser.add_argument("--input_folder_path", type=str, help="Path to the folder containing images.")
    parser.add_argument("--output_csv_path", type=str, help="Path to the output CSV file.")

    args = parser.parse_args()

    # Run the processing function
    process_images_in_folder(args.input_folder_path, args.output_csv_path)
