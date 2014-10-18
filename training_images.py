
from PIL import Image, ImageDraw, ImageFont
import shlex
import os

def createFont(fontFileName):
    fontfile = "/usr/share/fonts/truetype/msttcorefonts/%s" % fontFileName
    fnt = ImageFont.truetype(fontfile, 72)
    return fnt

def createImage(x, y, text, label, font, destDir):

    im = Image.new("RGB", (950, 150), "white")
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


courierFont = createFont("Courier_New.ttf")
timesFont = createFont("Times_New_Roman.ttf")
arialFont = createFont("Arial.ttf")

allFonts = [courierFont, timesFont, arialFont]

# the images used for training set will go here
training_images = "training_images"
if not os.path.exists(training_images):
    os.mkdir(training_images)

# the images used for test set will go here
test_images = "test_images"
if not os.path.exists(test_images):
    os.mkdir(test_images)


index = TrainingImagesIndex()
lines = [line.strip() for line in open('training_data.txt')]
for line in lines:

    if line == "":
        continue 

    text = line.upper()

    for font in allFonts:
        
        for x in [5,10,15,20]:
            for y in [15,25]:
                label = labelFromText(text)
                filename = createImage(x, y, text, label, font, training_images)
                index.addToIndex(filename, label)

    
index.saveIndex("ocr_training_images.txt")
    
