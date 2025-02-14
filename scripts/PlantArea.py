import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.cluster import KMeans

# Define the green color range in HSV
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

def calculate_green_percentage(image_path, binary_folder):
    """
    Calculate green percentage for an individual image.
    Returns a list of results for each box in the grid.
    """
    short_path = Path(*image_path.parts[-2:])
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
    grid_elements = []
    for contour in valid_contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) > 0:
            # Check if contour is better approximated by a circle
            # Calculate areas of contour, circle and rectangle
            contour_area = cv2.contourArea(contour)
            (x_circle, y_circle), radius = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * radius**2
            x, y, w, h = cv2.boundingRect(approx)
            rect_area = w * h
            
            # Compare which shape is closer to contour area
            circle_diff = abs(circle_area - contour_area)
            rect_diff = abs(rect_area - contour_area)
            if circle_diff < rect_diff:  # Circle is a better fit
                grid_elements.append([int(x_circle), int(y_circle), int(radius), int(radius), "circle"])  # Skip this contour as we can't properly process circular boxes
            else:
                if abs(w/h - 1) < 0.1:
                    grid_elements.append((x, y, w, h, "square"))
                elif abs(w/h - 2) < 0.1: # width approximately twice as big as height
                    grid_elements.append((x, y, w//2, h, "square"))
                    grid_elements.append((x+w//2, y, w//2, h, "square"))
                elif abs(w/h - 0.5) < 0.1: # height approximately twice as big as width
                    grid_elements.append((x, y, w, h//2, "square"))
                    grid_elements.append((x, y+h//2, w, h//2, "square"))
                else:
                    print(f"Warning: {short_path} has a box with an aspect ratio of {w/h}")
    grid_elements = sorted(grid_elements, key=lambda r: (round(r[1]/r[3]), round(r[0]/r[2])))
    # Calculate green percentage for each box
    results = []
    for i, element in enumerate(grid_elements):
        if element[4] == "circle":
            x, y, r = element[:3]
            box = image[max(0, y-r):min(y+r, image.shape[0]), max(0, x-r):min(x+r, image.shape[1])]
            print(x, y, r, box.shape)
            hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            total_pixels = np.pi * r**2
            green_pixels = cv2.countNonZero(green_mask)
            green_percentage = (green_pixels / total_pixels) * 100
            results.append((short_path, round(x/r), round(y/r), green_percentage, f"{i+1}/{len(grid_elements)}"))
        elif element[4] == "square":
            x, y, w, h  = element[:4]
            box = image[y:y+h, x:x+w]
            hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            total_pixels = w * h
            green_pixels = cv2.countNonZero(green_mask)
            green_percentage = (green_pixels / total_pixels) * 100
            results.append((short_path, round(x/w), round(y/h), green_percentage, f"{i+1}/{len(grid_elements)}"))
    coords = [el[:3] for el in results]
    if len(set(coords)) != len(coords):
        print(f"{short_path} coordinates detected incorrectly, check manually...", end=" ")
    
    ### Make a figure for the binary folder
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    plant_mask = cv2.inRange(hsv, lower_green, upper_green)
    box_mask = np.zeros_like(image)
    for i, element in enumerate(grid_elements):
        if element[4] == "circle":
            x, y, r = element[:3]
            # Create a circular mask
            Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
            dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
            circle_mask = dist_from_center <= r
            box_mask[circle_mask] = 1
            # Add border by removing 10px ring around circle
            outer_ring = (dist_from_center <= r+10) & (dist_from_center > r)
            box_mask[outer_ring] = 0
        elif element[4] == "square":
            x, y, w, h = element[:4]
            # Ensure indices are within bounds
            y_start = max(y-10, 0)
            y_end = min(y+h+10, box_mask.shape[0])
            x_start = max(x-10, 0)
            x_end = min(x+w+10, box_mask.shape[1])
            
            # Apply box mask with safe indices
            box_mask[min(y, box_mask.shape[0]):min(y+h, box_mask.shape[0]), 
                     min(x, box_mask.shape[1]):min(x+w, box_mask.shape[1])] = 1
            box_mask[y_start:y, x:x+w] = 0  
            box_mask[y+h:y_end, x:x+w] = 0  
            box_mask[y:y+h, x_start:x] = 0  
            box_mask[y:y+h, x+w:x_end] = 0  
    visualization = np.zeros_like(image)
    visualization[box_mask > 0] = 127
    visualization[plant_mask > 0] = 255
    cv2.imwrite(str(binary_folder / f"{short_path.stem}_detection.png"), visualization)
    ###
    return (results, 
            len(list(filter(lambda x: x[4] == "circle", grid_elements))), 
            len(list(filter(lambda x: x[4] == "square", grid_elements))))


def process_images_in_folder(input_folder, output_folder):
    """
    Process all images in a folder and save results into a single CSV file.
    """
    folder_path = Path(input_folder)
    output_csv = Path(output_folder) / "areas.tsv"
    if not folder_path.is_dir():
        raise NotADirectoryError(f"The specified folder does not exist: {folder_path}")

    # Initialize results list
    all_results = []

    # Iterate over all image files in the folder
    for image_path in folder_path.glob("*"):
        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            print(f"Processing image: {image_path}...", end=" ")
            results, circular_count, square_count = calculate_green_percentage(image_path, output_folder / "detection_vizualisations")
            all_results.extend(results)
            print("detected ", end=" ")
            if circular_count > 0:
                print(f"{circular_count} circular", end=" ")
            if square_count > 0 and circular_count > 0:
                print("and", end=" ")
            if square_count > 0:
                print(f"{square_count} rectangular", end=" ")
            print("boxes.")
        else:
            print(f"Skipping image: {image_path} (unsupported file extension)")
    
    # Save all results to a single CSV file
    if all_results:
        columns = ["filename", "column", "row", "percentage", "box#"]
        df = pd.DataFrame(all_results, columns=columns)
        df.to_csv(output_csv, index=False, sep="\t")
        print(f"Results saved to {output_csv}")
    else:
        print("No valid images were found in the folder.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate green percentage in grid images')
    parser.add_argument('--input_folder', type=str, help='Path to folder containing input images')
    parser.add_argument('--output_folder', type=str, help='Path to output folder')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output folder structure
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    binary_folder = output_folder / "detection_vizualisations"
    binary_folder.mkdir(exist_ok=True)
    
    # Process images
    input_folder = Path(args.input_folder)
    process_images_in_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
