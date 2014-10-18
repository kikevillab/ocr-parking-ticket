
from PIL import Image, ImageDraw, ImageFont
import shlex
import os


'''
Each training profile specifies what kind of images will be generated
'''
class TrainingProfile(object):

    def __init__(self, imgWidth, imgHeight, xPositions, yPositions, fontSize, trainingTextFile):
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight
        self.trainingTextFile = trainingTextFile
        self.xPositions = xPositions
        self.yPositions = yPositions
        self.fontSize = fontSize

    
trainParkingCodesSmall = TrainingProfile(190, 30, [0, 2, 4], [2, 5, 8], 12, 'training_data_sm.txt')
trainAlphabet = TrainingProfile(28, 28, [5], [0], 24, "training_data_alphabet.txt")
trainingProfile = trainAlphabet

def createFont(fontFileName, fontSize):
    fontfile = "/usr/share/fonts/truetype/msttcorefonts/%s" % fontFileName
    fnt = ImageFont.truetype(fontfile, fontSize)
    return fnt

def createImage(x, y, width, height, text, label, font, destDir):

    im = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(im)
    draw.text((x, y), text, font=font, fill="black")

    dirname = os.path.join(destDir, label)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    fontNameNoSpaces = getFontNameNoSpaces(font)    
    filename = os.path.join(dirname, "%s-%s-%s.png" % (fontNameNoSpaces, x, y))
    print filename
    if os.path.exists(filename):
        os.remove(filename)

    im.save(filename)

    return filename


'''
Given a font object (as returned from ImageFont.truetype()), get the font
name with any spaces removed from the name.  Eg, "Courier New" -> "CourierNew"
'''
def getFontNameNoSpaces(font):
    return font.getname()[0].replace(" ", "")


'''
Given text (eg, VND002 - Street Cleaning), return label (eg, VND002) to be
used for classification
'''
def labelFromText(text):
    tokens = shlex.split(text)
    parking_code = tokens[0]
    return parking_code

'''
A class to make it easier to save an index of all the training image files
'''
class TrainingImagesIndex(object):

    def __init__(self):
        self.indexEntries = []
        self.numericLabelMap = {}
        self.labelCounter = 0

    def addToIndex(self, filename, label):
        indexEntry = (filename, label)
        self.indexEntries.append(indexEntry)
        self.addLabelToMap(label)

    def addLabelToMap(self, label):
        # generate a map from labels to numbers and save it
        # eg, TRC7.2.87 -> 25 (since it's the 25th label we've seen)
        if not self.numericLabelMap.has_key(label):
            self.numericLabelMap[label] = self.labelCounter
            self.labelCounter += 1
    
    def saveNumericLabelMap(self, savefile):
        f = open(savefile, 'w')
        for label, numericLabel in self.numericLabelMap.iteritems():
            f.write("%s -> %s\n" % (label, numericLabel))
        f.close();

    def saveIndex(self, savefile):

        f = open(savefile, 'w')
        for indexEntry in self.indexEntries:
            filename = indexEntry[0]
            label = indexEntry[1]
            numericLabel = self.numericLabelMap[label]
            f.write("%s %s\n" % (filename, numericLabel))
        f.close();

        self.saveNumericLabelMap("ocr_training_labels.txt")


courierFont = createFont("Courier_New.ttf", trainingProfile.fontSize)
timesFont = createFont("Times_New_Roman.ttf", trainingProfile.fontSize)
arialFont = createFont("Arial.ttf", trainingProfile.fontSize)

allFonts = [courierFont, timesFont, arialFont]

# the images used for training set will go here
training_images = "training_images"
if not os.path.exists(training_images):
    os.mkdir(training_images)

# the images used for test set will go here
test_images = "test_images"
if not os.path.exists(test_images):
    os.mkdir(test_images)


lines = open(trainingProfile.trainingTextFile)  
index = TrainingImagesIndex()
lines = [line.strip() for line in lines]
imgWidth = trainingProfile.imgWidth
imgHeight = trainingProfile.imgHeight
for line in lines:

    if line == "":
        continue 

    text = line.upper()

    for font in allFonts:
        
        for x in trainingProfile.xPositions:
            for y in trainingProfile.yPositions:
                label = labelFromText(text)
                filename = createImage(x, y, imgWidth, imgHeight, text, label, font, training_images)
                index.addToIndex(filename, label)

    
index.saveIndex("ocr_training_images.txt")
    
