
from PIL import Image, ImageDraw, ImageFont
import shlex
import os

# this the font we'll use
fontfile = "/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf"
fnt = ImageFont.truetype(fontfile, 72)

# make a dest directory
destdir = "training_images"
if not os.path.exists(destdir):
    os.mkdir(destdir)

lines = [line.strip() for line in open('training_data.txt')]
for line in lines:

    if line == "":
        continue 

    text = line.upper()

    im = Image.new("RGB", (950, 150), "white")
    draw = ImageDraw.Draw(im)
    draw.text((10, 20), text, font=fnt, fill="black")
    tokens = shlex.split(line)
    parking_code = tokens[0]
    print parking_code

    dirname = os.path.join(destdir, parking_code)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    filename = os.path.join(dirname, "%s.png" % parking_code)
    if os.path.exists(filename):
        os.remove(filename)

    im.save(filename)
