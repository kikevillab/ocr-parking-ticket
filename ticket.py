#!/usr/bin/env python
# encoding: utf-8

import argparse
import glob
import os
import sys
from operator import itemgetter
from os.path import join, basename, dirname, abspath

try:
	from PIL import Image, ExifTags
	import numpy as np
	import cv2

except ImportError:
	print('Please install the required modules: numpy, Pillow, and cv2.')
	sys.exit()

class TicketApp(object):

	def __init__(self):
		"""App initialization.
		"""
		self.imgdir = dirname(abspath(__file__))
		self.filename = None
		self.debug = False

		# Parse the script arguments
		self.parse_args()

		# Remove any temporary images left in the dst dir
		self.remove_tmp_files()

		# Process the image
		img = self.load_image()
		img = self.segment_ticket(img)
		img = self.clusterize_colors(img)
		img = self.segment_largest_blob(img)
		img = self.segment_interest_row(img)

	def load_image(self):
		"""Load the input image and fix the Exif orientation.
		"""
		try:
			# Read the Exif data first
			filename = join(self.imgdir, self.filename)
			img = Image.open(filename)
			exif = img._getexif()

			# Destination temporary image
			dst = join(self.imgdir, 'step-1-original.png')

			if 274 in exif:
				orientation = exif.get(274)
				if orientation == 3:
					img = img.rotate(180, expand=True)
				elif orientation == 6:
					img = img.rotate(270, expand=True)
				elif orientation == 8:
					img = img.rotate(90, expand=True)
				img.save(dst, quality=100)

			img = cv2.imread(dst)	
			if not self.debug:
				os.remove(dst)
			return img

		except IOError:
			print('Cannot load the input image "%s"!' % filename)
			sys.exit()

	def segment_ticket(self, img):
		"""Segment the ticket object.
		"""
		if img is None:
			return

		# Create binary image
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		_, img_binary = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

		# Find the largest contour
		contours, _ = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)

		# Get rectangle approximation
		arclen = cv2.arcLength(contours[0], True)
		approx = cv2.approxPolyDP(contours[0], arclen * 0.02, True)
		contour = [approx]
		if len(approx) != 4:
			print('Failed segmenting rectangle. Exiting.')
			return

		# Create the mask image
		mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
		cv2.drawContours(mask, contour, 0, (255, 255, 255), -1)

		# Segment the ticket using the mask
		img_segmented = img & mask

		# Crop to the ticket blob
		x, y, width, height = cv2.boundingRect(contours[0])
		dst = img_segmented[y:y+height, x:x+width]

		# Set white background
		color = (255, 255, 255)
		cv2.floodFill(dst, None, (0, 0), color)
		cv2.floodFill(dst, None, (width-1, 0), color)
		cv2.floodFill(dst, None, (width-1, height-1), color)
		cv2.floodFill(dst, None, (0, height-1), color)
		dst = cv2.copyMakeBorder(dst, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=color)

		# Save image
		if self.debug:
			cv2.imwrite(join(self.imgdir, 'step-2-segment-ticket.png'), dst)

		return dst

	def clusterize_colors(self, img):
		"""Reduce colors by clusterizing the colors using K-Nearest algorithm.
		"""
		if img is None:
			return

		# Set the color classes
		colors = np.array([[0x00, 0x00, 0x00],
						   [0xff, 0xff, 0xff],
						   [0xff, 0x00, 0xff]], dtype=np.float32)
		classes = np.array([[0], [1], [2]], np.float32)

		# Predict with K-Nearest
		knn = cv2.KNearest()
		knn.train(colors, classes)
		img_flatten = np.reshape(np.ravel(img, 'C'), (-1, 3))
		retval, result, neighbors, dist = knn.find_nearest(img_flatten.astype(np.float32), 1)

		# Set new colors
		dst = colors[np.ravel(result, 'C').astype(np.uint8)]
		dst = dst.reshape(img.shape).astype(np.uint8)

		# Save image
		if self.debug:
			cv2.imwrite(join(self.imgdir, 'step-3-clusterize-colors.png'), dst)

		return dst

	def segment_largest_blob(self, img):
		"""Given an image with the ticket segmented, try to segment the table in the ticket.
		"""
		if img is None:
			return

		# Convert to binary image
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		_, img_binary = cv2.threshold(255-img_gray, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

		# Find contours
		contours, _ = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)

		# Keep the largest blob and remove the rest
		contours = contours[1:]
		cv2.drawContours(img, contours, -1, (255, 255, 255), -1)

		# Set the background to black
		height, width = img.shape[0], img.shape[1]
		cv2.floodFill(img, None, (0, 0), (0, 0, 0))
		cv2.floodFill(img, None, (width-1, 0), (0, 0, 0))
		cv2.floodFill(img, None, (width-1, height-1), (0, 0, 0))
		cv2.floodFill(img, None, (0, height-1), (0, 0, 0))

		# Save image
		if self.debug:
			cv2.imwrite(join(self.imgdir, 'step-4-segment-largest-blob.png'), img)

		return img

	def segment_interest_row(self, img):
		"""Segment the second row from the bottom of the table.
		"""
		if img is None:
			return

		# Create binary image
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		_, img_binary = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

		# Disconnect near blobs
		kernel = np.ones((3,3), np.uint8)
		img_eroded = cv2.erode(img_binary, np.ones((3,3), np.uint8), iterations=3)

		# Find the second row from the bottom of the table
		contours, _ = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contours = [c for c in contours if cv2.contourArea(c) > 1000]
		if len(contours) < 2:
			print('Failed segmenting the interest row.')
			return
		contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1], reverse=True)
		rect = cv2.boundingRect(contours[1])
		if (rect[2] < rect[3]):
			print('Failed locating the interest row.')
			return

		# Get and draw the rotated bounding rect
		rotated_rect = cv2.minAreaRect(contours[1])
		box = cv2.cv.BoxPoints(rotated_rect)
		box = np.int0(box)
		cv2.drawContours(img, [box], 0, (0, 0, 255), 10)

		# Save image
		cv2.imwrite(join(self.imgdir, 'result.png'), img)

		return img

	def remove_tmp_files(self):
		"""Delete the temporary files left on the dest dir.
		"""
		for filename in glob.glob('%s/step-*-*.png' % self.imgdir):
			os.remove(filename)

	def parse_args(self):
		parser = argparse.ArgumentParser(description='Script to read a ticket and segment interest text.')
		parser.add_argument('--debug', action='store_true', help='Keep the intermediate images when processing ticket image.')
		parser.add_argument('filename', help='The input image')
		args = parser.parse_args()

		# Check if the input image exists
		if not os.path.isfile(args.filename):
			print('Error: the input image "%s" does not exist.' % args.filename)
			sys.exit()

		# Assign the arguments as object attributes
		self.imgdir = dirname(abspath(args.filename))
		self.filename = basename(abspath(args.filename))
		self.debug = args.debug

if __name__ == "__main__":
	TicketApp()
