# Green area measurement

This is a script measuring the area filled by plants in boxes. It takes images as inputs, the images have to contain a grid m x n of white boxes on black background with plants growing inside. The script will output the % area occupied by plants in each box. 

## Installation


```bash
git clone https://github.com/Captain-Blackstone/PlantArea.git
cd PlantArea
pip3 install -r requirements.txt 
```

## Usage

The script runs in command line and has two arguments:
--input_folder_path: path to the folder containing images to be processed.
--output_csv_path: path to the output file

To test the installation you can run the example:
```bash
python3 scripts/PlantArea.py --input_folder_path ./images/ --output_csv_path ./test.csv
```

A file called "test.csv" is supposed to appear and have the following contents:
filename,row,column,percentage,box#
images/1001.jpg,0,0,3.2073001694132146,1/4
images/1001.jpg,0,1,2.87357114104692,2/4
images/1001.jpg,1,0,13.658094068626776,3/4
images/1001.jpg,1,1,12.268328570215896,4/4

filename: [parent directory]/[name of the processed file]
row, column: the script assumes the boxes containing the plants are organized in a rectangular shape. row and column refer to the position of the box being processed. row=0, column=0 is the top left corner.
percentage: percent of the box occupied by the plant.
box#: a somewhat redundant column, just reports the [index number of the current box]/[total number of boxes identified in the file].


