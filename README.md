Ticket Reader
=============
Python script to segment interest row on ticket images.


Ticket Table Contour Identification
-----------------------------------

- Ticket table features:
  - Aspect ratio of contour bounding box
  - Percent area of contour bounding box / total area of image
  - Total number of sub-contours (direct + indirect)
  - Total number of sub-contours that appear to be rectangles (via approxPolyDP)
  - Top-level sub-contour area ratio
    - Take the top-level sub-contours and add up the contourArea of each one
    - Calculate the ratio of the counter area / sub-contour's collective area
  - (optional) Number of sub-contours that have been identified as possible ticket table cells
    - Need a way to identify ticket table cells .. 
    - Bootstrap with number of sub-contours with aspect ratio in certain range
    - Eventually, replace this that uses a training algorithm
  - (advanced) Number of sub-contours that have been identified as containing text from Stroke Width Transform
- Machine learning training
  - One time: Identify actual ticket table coordinates in all test images	
  - For each image in training set
    - Generate feature vectors for each contour in the image
    - Each image has a pre-identified correct contour (via bounding box coords), which will be tagged as positive match
    - All other contours in image will be tagged as a negative match  
  - Use some sort of classifier to correctly classify ticket table contour
    - Start with hand classifier / if then decision tree
    - If needed, upgrade to OpenCV classifier or custom system
  - Verify against test set


Feature generation
------------------

For each contour
  - Is it 



Requirements
------------
To use the script, you need Python 2.7 and the following software installed on your computer.

- OpenCV (http://opencv.org) with Python support.
- Numpy module (http://numpy.org).
- Pillow module (https://pypi.python.org/pypi/Pillow)

To install OpenCV and Numpy, see the documentation on their site. To install Pillow, issue the following
command from your shell:

	pip install Pillow

Usage
-----
Once you have the requirement modules installed, you can run the script and pass the input image as the 
parameter like this:

	python ticket.py /path/to/image.png

The script will try to segment the interest row on the ticket. The output image is saved as `result.png` in
the same directory as the input image.

You can also pass the `--debug` flag:

	python ticket --debug /path/to/image.png

to see the temporary intermediate images created by the script. This is useful for debugging.

Known issues
------------
This script requires that the input image using a dark background. There is a chance that the script failed
to segment the interest row for poor input images. Please send me more input images to improve the accuracy
of the script.

