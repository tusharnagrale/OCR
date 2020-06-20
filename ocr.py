import re
import cv2
import numpy as np
import pytesseract
import mysql.connector


#loading image
image = cv2.imread("new.png")


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal:
def remove_noise(image):
    return cv2.medianBlur(image, 5)

# thresholding: converting image to binary form
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation: adding white pixel to the image at the boundary of black pixels
def dilate(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion: adding black pixel to the image at the boundary of white pixel so that the text will be sharpen
def erode(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation: not used because this is giving the blur image
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection : in this case because the text size in the image is very small this will not detect the edges properly
def canny(image):
    return cv2.Canny(image, 255, 255)



#calling functions
gray = get_grayscale(image)
thresh = thresholding(gray)
canny = canny(gray)
noise = remove_noise(gray)
erode = erode(thresh)
dilate = dilate(thresh)



#configuration to blacklist alphabets
custom_config = r'-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyz --psm 6'

text = pytesseract.image_to_string(thresh,config = custom_config,lang='eng')

#print(text)

text = text.replace(",","")

data = re.findall('\d*\.?\d+',text)
data = data[6],data[8],data[10],data[13]

print("extracted data from the image:")
print(data)


file = open("myfile.csv","w")
for ele in data:
    file.write(ele+',')
file.close()


#image conversions output
#cv2.imshow("gray",gray)

#cv2.imshow("thresh",thresh)

#cv2.imshow("canny",canny)

#cv2.imshow("erode",erode)

#cv2.imshow("dilate",dilate)

cv2.waitKey(0)

cv2.destroyAllWindows()



myconn = mysql.connector.connect(host = "localhost", user = "dbda",passwd = "dbda", database = "DIYCAM")
cur = myconn.cursor()

# creating database if not exist
cur.execute("create database IF NOT EXISTS DIYCAM")


#creating table if not exist
cur.execute("CREATE TABLE IF NOT EXISTS containers(max_gross DOUBLE NOT NULL,tare_weight DOUBLE NOT NULL,net DOUBLE NOT NULL,cu_cap DOUBLE NOT NULL)")

#listing tables in database to verify the table is created or not
print("\ntables in classwork databaes")

cur.execute("SHOW tables")
for table in cur:
      print(table)


query = "insert into containers(max_gross,tare_weight,net,cu_cap) values("+ str(data[0]) + "," + str(data[1]) + "," + str(data[2])+ "," + str(data[3])+");"

cur.execute(query)

myconn.commit()


print("\nextracted data from table")
cur.execute("select * from containers")

for x in cur:
  print(x)

myconn.close()











#import cv2
#import pytesseract
#from pytesseract import Output

#img = cv2.imread('/home/tushar/Downloads/oc.jpeg')

#d = pytesseract.image_to_data(img, output_type=Output.DICT)

#print(d.keys())

#n_boxes = len(d['text'])
#for i in range(n_boxes):
#    if int(d['conf'][i]) > 60:
#        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#cv2.imshow('img', img)
#cv2.waitKey(0)
