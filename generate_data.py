import argparse
import imageio
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os
import cv2


#Constants

#Folder constants
cardImageRoot='./cardsProcessed'
cardImageFilePrefix = 'card'
cardImageFileSuffix = '.png'
backRoot = "./backsProcessed"

#Generate image original size
cardOriginImagePath=cardImageRoot + "/" + cardImageFilePrefix + "0" + cardImageFileSuffix
img = Image.open(cardOriginImagePath)
imgOrigWidth = img.size[0]
imgOrigHeight = img.size[1]

#Generate background original size
backPath = '{0}/{1}.png'.format(backRoot, '{:06d}'.format(3))
back = Image.open(backPath)
backWidth = back.size[0]
backHeight = back.size[1]

#Scale range
scaleLow = 0.8
scaleHigh = 1.4

#Total number of backgrounds available
backChoiceHigh = 480
cardChoiceHigh = 3

#Whether to output grayscale images
outputMode = 'RGB'


#Utilities
def _change_lighting(image, offset=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if offset > 0:
        lim = 255 - offset
        v[v > lim] = 255
        v[(v <= lim) & (v>0)] += offset
    else:
        lim = 0 - offset
        cutoff = lim + 10
        v[(v <= cutoff) & (v>0)] = 10
        v[v > cutoff] -= lim

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def _salt_pepper (image):
    s_vs_p = 0.5
    amount = 0.01
    out = image
    # Generate Salt '1' noise
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image.shape]
    out[coords] = 255
    # Generate Pepper '0' noise
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in image.shape]
    out[coords] = 0
    return out
    
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

def _write_xml_file(xmlPath, noCard, imgPath, scale=None, pos1=None, pos2=None):
    with open(xmlPath, 'w') as xml_file:
        xml_file.write('<annotation>\n')
        xml_file.write('\t<noCard>{}</noCard>\n'.format(noCard))
        xml_file.write('\t<path>{}</path>\n'.format(imgPath))
        
        if noCard == 0:
        	xml_file.write('\t<scale>{}</scale>\n'.format(scale))
        	xml_file.write('\t<xmin>{}</xmin>\n'.format(pos1[0]))
        	xml_file.write('\t<ymin>{}</ymin>\n'.format(pos1[1]))
        	xml_file.write('\t<xmax>{}</xmax>\n'.format(pos2[0]))
        	xml_file.write('\t<ymax>{}</ymax>\n'.format(pos2[1]))
        else:
        	xml_file.write('\t<scale>{}</scale>\n'.format(0))
        	xml_file.write('\t<xmin>{}</xmin>\n'.format(0))
        	xml_file.write('\t<ymin>{}</ymin>\n'.format(0))
        	xml_file.write('\t<xmax>{}</xmax>\n'.format(0))
        	xml_file.write('\t<ymax>{}</ymax>\n'.format(0))        	
        	
        xml_file.write('</annotation>')
		
def _generate_empty_image(index):
	compPath = '{0}/{1}.png'.format(compRoot, '{:06d}'.format(index))
	
	#Pick a random background and make a copy
	backChoice = np.random.randint(1, high= backChoiceHigh)
	backPath = '{0}/{1}.png'.format(backRoot, '{:06d}'.format(backChoice))
	back = Image.open(backPath)
	if outputMode == 'L':
		back = back.convert('L')
	back.save(compPath)
	
	print("empty image path:", compPath)

	#generate the xml file
	xmlPath = '{0}/{1}.xml'.format(xmlRoot, '{:06d}'.format(index))
	_write_xml_file(xmlPath, 1, compPath)

def _generate_card_images(totalImages, maxLightOffset):
    print("Total Image count:", totalImages)
    print("Max lighting offset", maxLightOffset)

    maxRandomValue = 2*maxLightOffset + 1
    
    for i in range(totalImages):
        if np.random.randint(10)>8:
            _generate_empty_image(i)
            continue
            
        #Pick a random background
        backChoice = np.random.randint(1, high= backChoiceHigh)
        backPath = '{0}/{1}.png'.format(backRoot, '{:06d}'.format(backChoice))
        print("Background image path:", backPath)

        #Generate a random scale
        scale = scaleLow + np.random.random()*(scaleHigh-scaleLow)
        imgWidth = int(np.floor(scale * imgOrigWidth))
        imgHeight = int(np.floor(scale * imgOrigHeight))
        print("imgWidth:", imgWidth, "imgHeight: ", imgHeight)

        #generate a random translation
        posXHigh = backWidth-imgWidth
        posYHigh = backHeight-imgHeight
        posX = np.random.randint(1, high = posXHigh)
        posY = np.random.randint(1, high = posYHigh)
        print("posX:", posX, "posY: ", posY)


        cardImagePath='{0}/Card{1}.png'.format(cardImageRoot, '{:01d}'.format(np.random.randint(1, high= cardChoiceHigh)))
        print("Card image path:", cardImagePath)
        cardImage=Image.open(cardImagePath)
        
        #random blur the image
        if np.random.randint(1, high= 10) > 6: 
        	cardImage = cardImage.filter(ImageFilter.GaussianBlur(np.random.randint(1, high= 2)))
        
        #generating the random lighting offset
        offset=np.random.randint(0, maxRandomValue)-maxLightOffset
        print("Card lighting offset:", offset)
        cardArray = _change_lighting(np.asarray(cardImage), offset)
        
        #random salt and pepper
        if np.random.randint(1, high= 10) > 6: 
        	cardArray = _salt_pepper(cardArray)
        	
        cardImage = Image.fromarray(cardArray)
        
        
        comp = _create_composite_resize(cardImage, backPath, [imgWidth, imgHeight], [posX, posY])
        
        if outputMode == 'L': 
        	comp = comp.convert('L')
        	
        #generate the image file
        compPath = '{0}/{1}.png'.format(compRoot, '{:06d}'.format(i))
        print('compPath: ', compPath)
        comp.save(compPath)

        #generate the xml file
        xmlPath = '{0}/{1}.xml'.format(xmlRoot, '{:06d}'.format(i))
        _write_xml_file(xmlPath, 0, compPath, scale, [posX, posY], [posX+imgWidth, posY+imgHeight])
        
def parse_args():
    parser = argparse.ArgumentParser(description = 'Create training and validation data for card reader.')
    parser.add_argument('-t', '--totals', dest = 'totalImages', type=int, default=50, help = 'How many training data to generate?')
    parser.add_argument('-l', '--light', dest = 'maxLight', type=int, default=50, help = 'The upper limit of lighting offset?')
    parser.add_argument('-f', '--folder', dest = 'targetFolder', type=str, default='.', help = 'Where to put the generated images?')
    parser.add_argument('-m', '--mode', dest = 'mode', type=str, default='RGB', help = 'Grayscale output?')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
	args = parse_args()
	compRoot = args.targetFolder + '/images'
	os.makedirs(compRoot, exist_ok=True)
	xmlRoot = args.targetFolder + '/annotations'
	os.makedirs(xmlRoot, exist_ok=True)
	outputMode = args.mode
	
	_generate_card_images(args.totalImages, args.maxLight)
