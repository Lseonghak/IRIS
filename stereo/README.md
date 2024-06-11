# Stereo Image Processing and Deflection Analysis

This project processes stereo image pairs to compute disparity and depth maps, and perform deflection analysis.

## Features

- Load stereo parameters from a JSON file
- Rectify stereo images
- Compute disparity and depth maps
- Analyze deflections in identified regions

## Requirements

- Python 3.8 or higher
- OpenCV
- NumPy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/stereo-image-processing.git
    cd stereo-image-processing
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install opencv-python-headless numpy
    ```

## Usage

1. Prepare your stereo parameters in a `stereoParams.json` file.

2. Organize your project directory as follows:
    ```
    stereoParams.json
    data/
        left_images/
            1.jpg
            1.json
            2.jpg
            2.json
        right_images/
            1.jpg
            1.json
            2.jpg
            2.json
    ```

3. Run the main script:
    ```bash
    python main.py
    ```

4. The output images with disparity maps, depth maps, and deflection analysis will be saved in the `./output` directory.

## Example

Project directory structure:
'''
stereoParams.json
data/
    left_images/
        1.jpg
        1.json
        2.jpg
            2.json
    right_images/
        1.jpg
        1.json
        2.jpg
        2.json
'''