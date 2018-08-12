import argparse
import imageio
from PIL import Image, ImageDraw
import numpy as np
import os


#Constants

#Folder constants
imgPath = './cardsProcessed/corporate.png'
backRoot = "./backsProcessed"
compRoot = "./input_data/GeneratedData_Train"
xmlRoot = "./input_data/Annotations_Train"

#Generate image original size
img = Image.open(imgPath)
imgOrigWidth = img.size[0]
imgOrigHeight = img.size[1]

#Generate background original size
backPath = '{0}/{1}.png'.format(backRoot, '{:06d}'.format(3))
back = Image.open(backPath)
backWidth = back.size[0]
backHeight = back.size[1]

#Scale range
scaleLow = 0.8
scaleHigh = 1.0

#Total number of backgrounds available
backChoiceHigh = 3


#Utilities

def _create_mask(img, size): 

    img = img.resize(size=size)
    img_a = np.array(img.convert(mode='L'))
    img_a = img_a > 0
    return img, Image.fromarray(img_a.astype(np.uint8)*255)

def _create_composite_resize(img, backPath, size, pos):

    #create the image and mask with new size
    img, mask = _create_mask(img, size)
    
    #generate the image
    back = Image.open(backPath)
    back.paste(im=img, mask=mask, box=pos)
    
    return back

def _write_xml_file(xmlPath, imgPath, scale, pos1, pos2):
    with open(xmlPath, 'w') as xml_file:
        xml_file.write('<annotation>\n')
        xml_file.write('\t<path>{}</path>\n'.format(imgPath))
        xml_file.write('\t<scale>{}</scale>\n'.format(scale))
        xml_file.write('\t<xmin>{}</xmin>\n'.format(pos1[0]))
        xml_file.write('\t<ymin>{}</ymin>\n'.format(pos1[1]))
        xml_file.write('\t<xmax>{}</xmax>\n'.format(pos2[0]))
        xml_file.write('\t<ymax>{}</ymax>\n'.format(pos2[1]))

        xml_file.write('</annotation>')


def _generate_card_images(totalImages): 
    
    for i in range(totalImages):

        #Pick a random background
        backChoice = np.random.randint(1, high= backChoiceHigh)
        backPath = '{0}/{1}.png'.format(backRoot, '{:06d}'.format(backChoice))

        #Generate a random scale
        scale = scaleLow + np.random.random()*(scaleHigh-scaleLow)
        imgWidth = int(np.floor(scale * imgOrigWidth))
        imgHeight = int(np.floor(scale * imgOrigHeight))
        #print("imgWidth:", imgWidth, "imgHeight: ", imgHeight)

        #generate a random translation
        posXHigh = backWidth-imgWidth
        posYHigh = backHeight-imgHeight
        posX = np.random.randint(1, high = posXHigh)
        posY = np.random.randint(1, high = posYHigh)
        #print("posX:", posX, "posY: ", posY)

        #generate the image file
        compPath = '{0}/{1}.png'.format(compRoot, '{:06d}'.format(i))
        #print('compPath: ', compPath)
        comp = _create_composite_resize(img, backPath, [imgWidth, imgHeight], [posX, posY])
        comp.save(compPath)

        #generate the xml file
        xmlPath = '{0}/{1}.xml'.format(xmlRoot, '{:06d}'.format(i))
        _write_xml_file(xmlPath, compPath, scale, [posX, posY], [posX+imgWidth, posY+imgHeight])
        
def parse_args():
    parser = argparse.ArgumentParser(description = 'Create training and validation data for card reader.')
    parser.add_argument('-t', '--totals', dest = 'totalImages', type=int, default=50, help = 'How many training data to generate?')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
	args = parse_args()
	
	_generate_card_images(args.totalImages)
