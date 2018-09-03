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
scaleLow = 0.7
scaleHigh = 1.0

#Total number of backgrounds available
backChoiceHigh = 3
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

# LIZHI'S CODE
def _change_perspective(image, perspective, posX, posY, scale):
    imgHeight = image.shape[0]
    imgWidth = image.shape[1]

    # Original image's coordiates
    rect = np.array([
        [0, 0],
        [imgWidth, 0],
        [imgWidth, imgHeight],
        [0, imgHeight]], dtype="float32")
    print("orgi", rect)

    # Generate new width and height
    newWidth = int(_get_new_length(imgWidth, perspective))
    newHeight = int(_get_new_length(imgHeight, perspective))
    randomNum = np.random.uniform(1, perspective)

    print ("newHeightnewHeightnewHeight", newWidth, newHeight, randomNum)

    dst = np.array([
        [0, 0],
        [newWidth, randomNum],
        [newWidth, newHeight],
        [randomNum, newHeight]], dtype="float32")
    print("dest", dst)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (newWidth, newHeight))

    dstWithBackground = np.array([
        [posX, posY],
        [(newWidth + posX) * scale, (randomNum + posY) * scale],
        [(newWidth + posX) * scale, (newHeight + posY) * scale],
        [(randomNum + posX) * scale, (newHeight + posY) * scale]], dtype="float32")
    print("coord in background", dstWithBackground)

    # Return the warped image
    return warped, dstWithBackground

def _get_new_length(length, percentage):
    # Generate a random number between 1 and user input
    randomNum = np.random.uniform(1, percentage)
    # Generate positive or negtive
    posNeg = np.random.choice([-1, 1])
    randomNum = round(posNeg * randomNum, 2)
    print("posNeg", posNeg, "randomNum", randomNum)
    # Use the random number to generate new length or width for coordiates
    # newSize = length * (100 - randomNum)/100
    newSize = length
    return newSize

def _create_composite_resize(img, backPath, size, pos):

    #create the image and mask with new size
    img, mask = _create_mask(img, size)
    
    #generate the image
    back = Image.open(backPath)
    back.paste(im=img, mask=mask, box=pos)
    
    return back

def _write_xml_file(xmlPath, imgPath, scale, destArray):
    with open(xmlPath, 'w') as xml_file:
        xml_file.write('<annotation>\n')
        xml_file.write('\t<path>{}</path>\n'.format(imgPath))
        xml_file.write('\t<scale>{}</scale>\n'.format(scale))
        xml_file.write('\t<x0>{}</x0>\n'.format(int(destArray[0,0])))
        xml_file.write('\t<y0>{}</y0>\n'.format(int(destArray[0,1])))
        xml_file.write('\t<x1>{}</x1>\n'.format(int(destArray[1,0])))
        xml_file.write('\t<y1>{}</y1>\n'.format(int(destArray[1,1])))
        xml_file.write('\t<x2>{}</x2>\n'.format(int(destArray[2,0])))
        xml_file.write('\t<y2>{}</y2>\n'.format(int(destArray[2,1])))
        xml_file.write('\t<x3>{}</x3>\n'.format(int(destArray[3,0])))
        xml_file.write('\t<y3>{}</y3>\n'.format(int(destArray[3,1])))

        xml_file.write('</annotation>')


def _generate_card_images(totalImages, maxLightOffset, perspective):
    print("Total Image count:", totalImages)
    print("Max lighting offset", maxLightOffset)

    maxRandomValue = 2*maxLightOffset + 1
    
    for i in range(totalImages):

        #Pick a random background
        backChoice = np.random.randint(1, high= backChoiceHigh)
        backPath = '{0}/{1}.png'.format(backRoot, '{:06d}'.format(backChoice))
        print("Background image path:", backPath)

        #Generate a random scale
        # scale = scaleLow + np.random.random()*(scaleHigh-scaleLow)
        scale = 1
        imgWidth = int(np.floor(scale * imgOrigWidth))
        imgHeight = int(np.floor(scale * imgOrigHeight))
        print("scale", scale, "imgWidth:", imgWidth, "imgHeight: ", imgHeight)

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
        
        #adding perspective
        cv_comp = np.asarray(cardImage).copy()
        warp, dstWithBackground = _change_perspective(cv_comp, perspective, posX, posY, scale)

        comp = Image.fromarray(warp)
        comp = _create_composite_resize(comp, backPath, [imgWidth, imgHeight], [posX, posY])
        
        if outputMode == 'L': 
        	comp = comp.convert('L')
        	
        #generate the image file
        compPath = '{0}/{1}.png'.format(compRoot, '{:06d}'.format(i))
        print('compPath: ', compPath)
        comp.save(compPath)

        #generate the xml file
        xmlPath = '{0}/{1}.xml'.format(xmlRoot, '{:06d}'.format(i))
        _write_xml_file(xmlPath, compPath, scale, dstWithBackground)
        
def parse_args():
    parser = argparse.ArgumentParser(description = 'Create training and validation data for card reader.')
    parser.add_argument('-t', '--totals', dest = 'totalImages', type=int, default=50, help = 'How many training data to generate?')
    parser.add_argument('-l', '--light', dest = 'maxLight', type=int, default=50, help = 'The upper limit of lighting offset?')
    parser.add_argument('-f', '--folder', dest = 'targetFolder', type=str, default='.', help = 'Where to put the generated images?')
    parser.add_argument('-m', '--mode', dest = 'mode', type=str, default='RGB', help = 'Grayscale output?')
    parser.add_argument('-p', '--perspective', dest='perspective', type=int, default=10, help='max perspective scale')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
	args = parse_args()
	compRoot = args.targetFolder + '/images'
	os.makedirs(compRoot, exist_ok=True)
	xmlRoot = args.targetFolder + '/annotations'
	os.makedirs(xmlRoot, exist_ok=True)
	outputMode = args.mode
	
	_generate_card_images(args.totalImages, args.maxLight, args.perspective)
