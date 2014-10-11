
from PIL import Image, ImageDraw, ImageFont
import shlex
import os


def createFont(fontFileName):
    fontfile = "/usr/share/fonts/truetype/msttcorefonts/%s" % fontFileName
    fnt = ImageFont.truetype(fontfile, 72)
    return fnt

courierFont = createFont("Courier_New.ttf")
timesFont = createFont("Times_New_Roman.ttf")

trainingFonts = [courierFont]
testFonts = [timesFont] 
allFonts = testFonts + trainingFonts

# the images used for training set will go here
training_images = "training_images"
if not os.path.exists(training_images):
    os.mkdir(training_images)

# the images used for test set will go here
test_images = "test_images"
if not os.path.exists(test_images):
    os.mkdir(test_images)


lines = [line.strip() for line in open('training_data.txt')]
for line in lines:

    if line == "":
        continue 

    text = line.upper()

    for font in allFonts:

        destDir = "error"
        if font in trainingFonts:
            destDir = training_images
        else:
            destDir = test_images

        im = Image.new("RGB", (950, 150), "white")
        draw = ImageDraw.Draw(im)
        draw.text((10, 20), text, font=font, fill="black")
        tokens = shlex.split(line)
        parking_code = tokens[0]
        print parking_code

        dirname = os.path.join(destDir, parking_code)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        filename = os.path.join(dirname, "%s.png" % font.getname()[0])
        if os.path.exists(filename):
            os.remove(filename)

        im.save(filename)

    
