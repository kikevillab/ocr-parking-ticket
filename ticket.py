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
		img = self.extract_contour_features(img)
		#img = self.clusterize_colors(img)
		#img = self.segment_largest_blob(img)
		#img = self.segment_interest_row(img)

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

	def extract_contour_features(self, img):
		"""
                Extract the features.  One feature row for each contour.
		"""
		if img is None:
			return

		# Create binary image
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# _, img_binary = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                img_binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)

		# Save image
		if self.debug:
			cv2.imwrite(join(self.imgdir, 'step-2a-segment-ticket.png'), img_binary)
                

		# Find contours
                contours, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		# contours = sorted(contours, key=cv2.contourArea, reverse=True)
                hierarchy = hierarchy[0]

                i = 0
                for contour in contours:
                        contourHierarchy = hierarchy[i]
                        print("hierarchy: " + str(i) + " -> " + str(contourHierarchy))

                        arclen = cv2.arcLength(contour, True)
                        if arclen < 100:
                                i += 1
                                continue
                        features = self.getFeatures(img_binary, contour, i, hierarchy)
                        rotated_rect = cv2.minAreaRect(contour)
                        box = cv2.cv.BoxPoints(rotated_rect)
                        box = np.int0(box)
                        print("rotated_rect: " + str(rotated_rect))
                        
                        # Get rectangle approximation
                        arclen = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, arclen * 0.02, True)
                        approxInner = [approx]
                        print("approxInner: " + str(approxInner))
                        printable = [5423, 5424, 5425, 5426, 5427, 5428, 5429, 5430, 5431, 5432, 5433, 5434, 5435, 5436, 5437, 5438, 5439, 5440, 5441, 5442, 5443, 5444, 5445, 5446, 5447, 5448, 5449, 5450, 5451, 5452, 5453, 5454, 5455, 5456, 5457, 5458, 5459, 5460, 5461, 5462, 5463, 5464, 5465, 5466, 5467, 5468, 5469, 5470, 5471, 5472, 5473, 5474, 5475, 5476, 5477, 5478, 5479, 5480, 5481, 5482, 5483, 5484, 5485, 5486, 5487, 5488, 5489, 5490, 5491, 5492, 5493, 5494, 5495, 5496, 5497, 5498, 5499, 5500, 5501, 5502, 5503, 5504, 5505, 5506, 5507, 5508, 5509, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525, 5526, 5527, 5528, 5529, 5530, 5531, 5532, 5533, 5534, 5535, 5536, 5537, 5538, 5539, 5540, 5541, 5542, 5543, 5544, 5545, 5546, 5547, 5548, 5549, 5550, 5551, 5552, 5553, 5554, 5555, 5556, 5557, 5558, 5559, 5560, 5561, 5562, 5563, 5564, 5565, 5566, 5567, 5568, 5569, 5570, 5571, 5572, 5573, 5574, 5575, 5576, 5577, 5578, 5579, 5580, 5581, 5582, 5583, 5584, 5585, 5586, 5587, 5588, 5589, 5590, 5591, 5592, 5593, 5594, 5595, 5596, 5597, 5598, 5599, 5600, 5601, 5602, 5603, 5604, 5605, 5606, 5607, 5608, 5609, 5610, 5611, 5612, 5613, 5614, 5615, 5616, 5617, 5618, 5619, 5620, 5621, 5622, 5623, 5624, 5625, 5626, 5627, 5628, 5629, 5630, 5631, 5632, 5633, 5634, 5635, 5636, 5637, 5638, 5639, 5640, 5641, 5642, 5643, 5644, 5645, 5646, 5647, 5648, 5649, 5650, 5651, 5652, 5653, 5654, 5655, 5656, 5657, 5658, 5659, 5660, 5661, 5662, 5663, 5664, 5665, 5666, 5667, 5668, 5669, 5670, 5671, 5672, 5673, 5674, 5675, 5676, 5677, 5678, 5679, 5680, 5681, 5682, 5683, 5684, 5685, 5686, 5687, 5688, 5689, 5690, 5691, 5692, 5693, 5694, 5695, 5696, 5697, 5698, 5699, 5700]

                        if i in printable:
                                cv2.drawContours(img, [box], 0, (0, 0, 255), 10)
                        i += 1

                cv2.imwrite(join(self.imgdir, 'step-2b-segment-ticket.png'), img)

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

        def getFeatures(self, img_binary, contour, contour_idx, hierarchy):
                """
                Get all the features for this contour:
                - Aspect ratio of contour bounding box
                - Percent area of contour bounding box / total area of image
                - Total number of sub-contours (direct + indirect)
                - Total number of sub-contours that appear to be rectangles (via approxPolyDP)
                - Top-level sub-contour area ratio
                - Take the top-level sub-contours and add up the contourArea of each one
                - Calculate the ratio of the counter area / sub-contour's collective area
                """
                aspectRatio = self.getFeatureAspectRatio(contour)
                print("aspectRatio: %s" % aspectRatio)
                percent_area = self.getFeaturePercentArea(img_binary, contour)
                print("percentArea: %s" % percent_area)
                subContours = self.findNumberChildContours(contour_idx, hierarchy)
                numSubContours = len(subContours)
                print("numSubContours of %s: %s" % (contour_idx, numSubContours))
                print("subContours : %s" % subContours)
                return 

        def findNumberChildContours(self, contour_idx, hierarchy):
                print("findNumberChildContours called with contour: %s" % contour_idx)
                accumulator = []
                contour_hierarchy = hierarchy[contour_idx]
                _, _, first_child, _ = contour_hierarchy
                if first_child != -1:
                        self.findSiblingsAndChildren(accumulator, hierarchy, first_child)
                return accumulator

        def findSiblingsAndChildren(self, accumulator, hierarchy, contour_idx):
                print("findNumberChildContours called with contour: %s accum len: %s" % (contour_idx, len(accumulator)))
                
                print("append: %s" % contour_idx)
                accumulator.append(contour_idx)

                # [Next, Previous, First_Child, Parent]
                contour_hierarchy = hierarchy[contour_idx]
                nxt, prev, first_child, parent = contour_hierarchy
                if first_child != -1:
                        self.findSiblingsAndChildren(accumulator, hierarchy, first_child)
                
                while nxt != -1:
                      print("append: %s" % nxt)
                      accumulator.append(nxt)
                      nxt, _, first_child_nxt, _ = hierarchy[nxt]
                      if first_child_nxt != -1:
                              self.findSiblingsAndChildren(accumulator, hierarchy, first_child_nxt)
                      
                      


        def getFeaturePercentArea(self, img_binary, contour):
                """
                Percent area of contour bounding box / total area of image
                """
                rotated_rect = cv2.minAreaRect(contour)
                center, size, angle = rotated_rect
                width, height = size
                contour_area = width * height
                img_binary_cvmat = cv2.cv.fromarray(img_binary)
                img_width, img_height = cv2.cv.GetSize(img_binary_cvmat)
                img_area = img_width * img_height
                percent_area = contour_area / img_area
                print("percent_area: %s" % (percent_area))
                return percent_area

        def getFeatureAspectRatio(self, contour):
                """
                Get the aspect ratio of this contour
                """
                rotated_rect = cv2.minAreaRect(contour)
                center, size, angle = rotated_rect
                width, height = size
                aspectRatio = width / height  

                return aspectRatio
                

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
