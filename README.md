# Green area measurement

This is a program measuring the area filled by plants in boxes. It takes images as inputs, the images have to contain a grid m x n of white boxes on black background with plants growing inside. The script will output the % area occupied by plants in each box. 

## Installation


```bash
git clone https://github.com/Captain-Blackstone/PlantArea.git
cd PlantArea
pip3 install -r requirements.txt 
```

## Usage

The script runs in command line and has two arguments: <br />
--input_folder: path to the folder containing images to be processed. <br />
--output_folder: path to the output folder. The output folder will contain a table (areas.tsv) and a folder (detection_vizualisations). The folder contains binarized (trinarized?) images with background in black, box in grey and plants in white and is useful for manual inspection (to make sure the areas were detected correctly). The table contains the results of the area detaction and is discussed later.<br />

To test the installation you can run the example:
```bash
python3 scripts/PlantArea.py --input_folder ./images/ --output_folder testOut
```

A folder called "testOut" is supposed to appear and contain a subfolder with detection visualization and the table areas.tsv with the following content: 
```
filename,column,row,percentage,box#
images/1001.jpg,0,0,2.87357114104692,1/4
images/1001.jpg,1,0,3.2073001694132146,2/4
images/1001.jpg,0,1,13.658094068626776,3/4
images/1001.jpg,1,1,12.268328570215896,4/4
```
filename: [parent directory]/[name of the processed file] <br />
row, column: the script assumes the boxes containing the plants are organized in a rectangular shape. row and column refer to the position of the box being processed. row=0, column=0 is the top left corner. <br />
percentage: percent of the box occupied by the plant. <br />
box#: a somewhat redundant column, just reports the [index number of the current box]/[total number of boxes identified in the file]. <br />


