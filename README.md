Ticket Reader
=============
Python script to segment interest row on ticket images.

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

Contact the author
------------------
Send bug reports to: nash@bsd-noobz.com
