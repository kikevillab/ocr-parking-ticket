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
                self.shouldRunTests = False

		# Parse the script arguments
		self.parse_args()

		# Remove any temporary images left in the dst dir
		self.remove_tmp_files()

                if self.shouldRunTests == True:
                        self.runTests()


        def processImage(self):
		
                # Process the image
		img = self.load_image()
		ticketTableContours = self.findTicketTableContourCandidates(img)
                for contour in ticketTableContours:
                        self.drawContour(contour, img)       

                if self.debug:
                        cv2.imwrite('step-process-image.png', img)


	def load_image(self):
		"""Load the input image and fix the Exif orientation.
		"""
		try:
			# Read the Exif data first
			filename = join(self.imgdir, self.filename)
			img = Image.open(filename)
			exif = img._getexif()

			# Destination temporary image
			dst = 'step-1-original.png'

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

	def findTicketTableContourCandidates(self, img):

		"""
                Find the contours that are most likely to be the one that corresponds
                to the ticket table.  Most likely result is at 0th index of returned list.
		"""
		if img is None:
			return

                results = []

		# Create binary image
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# _, img_binary = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                img_binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)

		# Save image
		if self.debug:
			cv2.imwrite('step-find-ticket-table-contour.png', img_binary)

		# Find contours
                contours, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                hierarchy = hierarchy[0]

                i = 0
                for contour in contours:

                        contourHierarchy = hierarchy[i]

                        features = self.getFeatures(img_binary, contours, contour, i, hierarchy)
                        
                        aspectRatio, percent_area, numSubContours, numRectSubContours, subContourAreaRatio, immediateSubcontourRatio = features

                        contourLength = cv2.arcLength(contour, True)
                        approxPolyContour = cv2.approxPolyDP(contour, 0.02 * contourLength, True)


                        # this works for tickets 0-7
                        # if numSubContours > 100 and numRectSubContours > 20 and percent_area > 0.05 and percent_area < 0.38 and len(approxPolyContour) == 4 and cv2.isContourConvex(approxPolyContour) and subContourAreaRatio > 0.75 and immediateSubcontourRatio > 0.05:
                        
                        # this works for ticket 8
                        if numSubContours > 100 and numRectSubContours > 20 and subContourAreaRatio > 0.75 and immediateSubcontourRatio > 0.05 and percent_area < 0.38 and len(approxPolyContour) == 4 and cv2.isContourConvex(approxPolyContour) and percent_area < 0.38:

                                print("\n\ncontour %s:\n\n" % i)
                                print("aspectRatio: %s" % aspectRatio)
                                print("percentArea: %s" % percent_area)
                                print("numSubContours of %s: %s" % (i, numSubContours))
                                print("rect subContours : %s" % numRectSubContours)
                                print("subContourAreaRatio : %s" % subContourAreaRatio)
                                print("immediateSubcontourRatio: %s" % immediateSubcontourRatio)

                                results.append(contour)
                        
                        i += 1

                return results


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
			cv2.imwrite(join(self.imgdir, 'step-clusterize-colors.png'), dst)
 
		return dst


        def drawContour(self, contour, img):
                rotated_rect = cv2.minAreaRect(contour)
                box = cv2.cv.BoxPoints(rotated_rect)
                box = np.int0(box)
                cnt_len = cv2.arcLength(contour, True)
                cnt = cv2.approxPolyDP(contour, 0.02*cnt_len, True)
                cv2.drawContours(img, [cnt], 0, (0, 0, 255), 10)        

        def drawContourAndImmediateChildren(self, contour, img, contour_idx, hierarchy, contours):

                # draw the contour itself
                self.drawContour(contour, img)

                # get all immediate child contours and draw each one
                childContourIndexes = self.getContourIndexesImmediateChildren(contour, contour_idx, hierarchy, contours)
                for childContourIdx in childContourIndexes:
                        contourToDraw = contours[childContourIdx]
                        self.drawContour(contourToDraw, img)


        def getContourIndexesImmediateChildren(self, contour, contour_idx, hierarchy, contours):
                children = []
                contour_hierarchy = hierarchy[contour_idx]
                _, _, first_child_idx, _ = contour_hierarchy
                if first_child_idx != -1:
                        children.append(first_child_idx)
                        nextChildContourIdx = first_child_idx
                        while nextChildContourIdx != -1:
                                contour_hierarchy = hierarchy[nextChildContourIdx]
                                nextChildContourIdx, _, _, _ = contour_hierarchy
                                if nextChildContourIdx != -1:
                                        children.append(nextChildContourIdx)
                return children

                

        def getFeatures(self, img_binary, contours, contour, contour_idx, hierarchy):
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

                """
                Problematic Ticket6 ideas:
                - Draw actual contour and look at shape, see how convex it is or other properties to differentiate
                - Do top-level sub-contour area ratio as described above
                """
                aspectRatio = self.getFeatureAspectRatio(contour)
                percent_area = self.getFeaturePercentArea(img_binary, contour)
                subContours = self.findChildContours(contour_idx, hierarchy)
                numSubContours = len(subContours)
                rectSubContours = self.filterRectangleContours(contours, subContours)
                numRectSubContours = len(rectSubContours)
                subContourAreaRatio = self.subContourAreaRatio(contour, contour_idx, hierarchy, contours)
                immediateSubcontourRatio = self.getFeatureImmeditateSubcontourRatio(contour, contour_idx, hierarchy, contours, subContours)

                return [aspectRatio, percent_area, numSubContours, numRectSubContours, subContourAreaRatio, immediateSubcontourRatio]


        def getFeatureImmeditateSubcontourRatio(self, contour, contour_idx, hierarchy, contours, subContours):
                """
                Ratio of number of immediate subcontours over the number of total subcontours
                Eg, ticket7.jpg distinguish between ticket table and ticket inner contour
                """
                numImmediateSubcontours = len(self.getContourIndexesImmediateChildren(contour, contour_idx, hierarchy, contours))
                #print("numImmediateSubcontours: %s", numImmediateSubcontours)
                #print("len(subContours): %s", len(subContours))
                if len(subContours) > 0:
                        return numImmediateSubcontours*1.0 / len(subContours)*1.0
                else:
                        return 0

        def subContourAreaRatio(self, contour, contour_idx, hierarchy, contours):
                
                # find all immediate child contours
                childContourIndexes = self.getContourIndexesImmediateChildren(contour, contour_idx, hierarchy, contours)
                
                # get area of each one and add it to running total
                childContoursArea = 0.0
                for childContourIdx in childContourIndexes:
                        childContour = contours[childContourIdx]
                        childContourArea = self.contourArea(childContour)
                        childContoursArea += childContourArea

                parentContourArea = self.contourArea(contour)

                if parentContourArea > 0:
                        return childContoursArea / parentContourArea
                else:
                        return 0

 
        def contourArea(self, contour):
                rotated_rect = cv2.minAreaRect(contour)
                center, size, angle = rotated_rect
                width, height = size
                contour_area = width * height
                return contour_area

        def filterRectangleContours(self, contours, subContours):
                def is_rectangle(contour_idx):
                        contour = contours[contour_idx]
                        arclen = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, arclen * 0.02, True)
                        approxInner = [approx]
                        if len(approx) == 4:
                                return contour_idx

                return filter(is_rectangle, subContours)

        def findChildContours(self, contour_idx, hierarchy):
                accumulator = []
                contour_hierarchy = hierarchy[contour_idx]
                _, _, first_child, _ = contour_hierarchy
                if first_child != -1:
                        self.findSiblingsAndChildren(accumulator, hierarchy, first_child)
                return accumulator

        def findSiblingsAndChildren(self, accumulator, hierarchy, contour_idx):

                accumulator.append(contour_idx)

                # [Next, Previous, First_Child, Parent]
                contour_hierarchy = hierarchy[contour_idx]
                nxt, prev, first_child, parent = contour_hierarchy
                if first_child != -1:
                        self.findSiblingsAndChildren(accumulator, hierarchy, first_child)
                
                while nxt != -1:
                      accumulator.append(nxt)
                      nxt, _, first_child_nxt, _ = hierarchy[nxt]
                      if first_child_nxt != -1:
                              self.findSiblingsAndChildren(accumulator, hierarchy, first_child_nxt)


        def getFeaturePercentArea(self, img_binary, contour):
                """
                Percent area of contour bounding box / total area of image
                """
                contour_area = self.contourArea(contour)
                img_area = self.getImageArea(img_binary)
                percent_area = contour_area / img_area
                return percent_area

        def getImageArea(self, img_binary):
                img_binary_cvmat = cv2.cv.fromarray(img_binary)
                img_width, img_height = cv2.cv.GetSize(img_binary_cvmat)
                img_area = img_width * img_height
                return img_area


        def getFeatureAspectRatio(self, contour):
                """
                Get the aspect ratio of this contour
                """
                rotated_rect = cv2.minAreaRect(contour)
                center, size, angle = rotated_rect

                        
                width, height = size
                if height > 0:
                        if abs(angle) < 1:
                                aspectRatio = width / height  
                        elif abs(angle) > 85 or abs(angle) < 95:
                                aspectRatio = height / width  
                        else:
                                raise Exception("Unexpected aspect ratio")
                else:
                        aspectRatio = 0

                return aspectRatio


	def remove_tmp_files(self):
		"""Delete the temporary files left on the dest dir.
		"""
		for filename in glob.glob('%s/step-*-*.png' % self.imgdir):
			os.remove(filename)

	def parse_args(self):
		parser = argparse.ArgumentParser(description='Script to read a ticket and segment interest text.')
		parser.add_argument('--debug', action='store_true', help='Keep the intermediate images when processing ticket image.')
		parser.add_argument('--tests', action='store_true', help='Run tests.')
		parser.add_argument('filename', help='The input image')
		args = parser.parse_args()

		# Check if the input image exists
		if not args.tests and not os.path.isfile(args.filename):
			print('Error: the input image "%s" does not exist.' % args.filename)
			sys.exit()

		# Assign the arguments as object attributes
		self.imgdir = dirname(abspath(args.filename))
		self.filename = basename(abspath(args.filename))
		self.debug = args.debug
                self.shouldRunTests = args.tests


        def runTests(self):

                # - For each test ticket (eg, ticket0.jpg) 
                #   - Read expected result file to get expected coords of ticket table (eg, read ticket0.json, get ticketTableContour field)
                #   - Call findTicketTableContourCandidates with ticket0.jpg to get ticket table contour
                #   - Compare expected vs actual, fail if no match

                import os

                print("runTests ..")
                def verifyCorrectTicketTables(arg, dirname, fnames):
                        print("verifyCorrectTicketTables ..")
                        for fname in fnames:
                                if os.path.isdir(fname):
                                        print("Skipping directory: %s" % fname)
                                        continue
                                print(fname)
                                verifyCorrectTicketTable(dirname, fname)

                def verifyCorrectTicketTable(dirname, fname):
                        
                        print("-" * 100)
                        print("verifyCorrectTicketTable for: %s" % fname)
                        print("-" * 100)

                        #if fname != "ticket8-training.jpg":
                        #        return 

                        self.imgdir = dirname
                        self.filename = fname
                        img = self.load_image()

                        ticketTableContours = self.findTicketTableContourCandidates(img)


                        contour = ticketTableContours[0]
                        if len(ticketTableContours) != 1:
                                raise Exception("Expected 1 result, got: %s" % len(ticketTableContours))

                        print("\n\nbest contour: %s" % contour)
                        rotated_rect = cv2.minAreaRect(contour)
                        rotated_rect_center = np.int0(rotated_rect[0])
                        rotated_rect_size = np.int0(rotated_rect[1])
                        print("rotated_rect_center: %s" % str(rotated_rect_center))
                        print("rotated_rect_size: %s" % str(rotated_rect_size))
                        box = cv2.cv.BoxPoints(rotated_rect)
                        print("box: %s" % str(box))
                        box = np.int0(box)
                        print("box: %s" % str(box))

                        if self.debug:
                                #for contour in ticketTableContours:
                                #        self.drawContour(contour, img)       
                                self.drawContour(contour, img)       
                                cv2.imwrite('%s-ticket-table.png' % fname, img)

                        if fname == "ticket0-training.jpg":
                                expectedCenter = np.int0((1078, 2379))
                                expectedSize = np.int0((946, 1530))
                        elif fname == "ticket1-training.jpg":
                                expectedCenter = np.int0((1163, 1965))
                                expectedSize = np.int0((865, 1376))
                        elif fname == "ticket2-training.jpg":
                                expectedCenter = np.int0((1111, 1950))
                                expectedSize = np.int0((803, 1228))
                        elif fname == "ticket3-training.jpg":
                                expectedCenter = np.int0((1104, 2088))
                                expectedSize = np.int0((1113, 699))
                        elif fname == "ticket4.jpg":
                                expectedCenter = np.int0((1204, 2028))
                                expectedSize = np.int0((900, 576))
                        elif fname == "ticket5-training.jpg":
                                expectedCenter = np.int0((1158, 1557))
                                expectedSize = np.int0((986, 1457))
                        elif fname == "ticket6-training.jpg":
                                expectedCenter = np.int0((1289, 1991))
                                expectedSize = np.int0((1339, 854))
                        elif fname == "ticket7.jpg":
                                expectedCenter = np.int0((1049, 1661))
                                expectedSize = np.int0((520, 817))
                        elif fname == "ticket8-training.jpg":
                                expectedCenter = np.int0((1231, 2013))
                                expectedSize = np.int0((485, 764))
                        elif fname == "ticket9.jpg":
                                expectedCenter = np.int0((1177, 1988))
                                expectedSize = np.int0((1273, 1888))

                        if rotated_rect_center[0] != expectedCenter[0] or rotated_rect_center[1] != expectedCenter[1]:
                                msg = "Actual center (%s) did not match expected (%s)" % (rotated_rect[0], expectedCenter)
                                raise Exception(msg)
                        if rotated_rect_size[0] != expectedSize[0] or rotated_rect_size[1] != expectedSize[1]:
                                msg = "Actual size (%s) did not match expected (%s)" % (rotated_rect[0], expectedSize)
                                raise Exception(msg)

                # Process all test + training images
                imagesDir = "data"
                os.path.walk(imagesDir, verifyCorrectTicketTables, None)

                # >>> import json
                #>>> json.loads('["foo", {"bar":["baz", null, 1.0, 2]}]')
                #['foo', {'bar': ['baz', None, 1.0, 2]}]



if __name__ == "__main__":
	TicketApp()
