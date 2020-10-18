import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"D:\JD\Tesseract-OCR\tesseract.exe"

# 1. Image PreProcessing ( closing operation )

img = cv2.imread('sudoku.png')
img = cv2.GaussianBlur(img,(5,5),0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mask = np.zeros((gray.shape),np.uint8)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
div = np.float32(gray)/(close)
res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)

#2. Finding Sudoku Square and Creating Mask Image

thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)
contour,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
best_cnt = None
for cnt in contour:
    area = cv2.contourArea(cnt)
    if area > 1000:
        if area > max_area:
            max_area = area
            best_cnt = cnt

cv2.drawContours(mask,[best_cnt],0,255,-1)
cv2.drawContours(mask,[best_cnt],0,0,3)

res= cv2.bitwise_and(res,mask)
cv2.imshow('res',res)

#3. Finding Vertical lines

kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

dx = cv2.Sobel(res,cv2.CV_16S,1,0)
dx = cv2.convertScaleAbs(dx)
cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if h/w > 5:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)
close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
closex = close.copy()

#4. Finding Horizontal Lines
kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
dy = cv2.Sobel(res,cv2.CV_16S,0,2)
dy = cv2.convertScaleAbs(dy)
cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if w/h > 5:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)

close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
closey = close.copy()

cv2.imshow('closex',closex)
cv2.imshow('closey',closey)
#5. Finding Grid Points

res = cv2.bitwise_and(closex,closey)

cv2.imshow('res',res)

#6. Correcting the defects

contour, hier = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
centroids = []
for cnt in contour:
    mom = cv2.moments(cnt)
    (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
    cv2.circle(img,(x,y),4,(0,255,0),-1)
    centroids.append((x,y))

centroids = np.array(centroids,dtype = np.float32)
c = centroids.reshape((100,2))
c2 = c[np.argsort(c[:,1])]

b = np.vstack([c2[i*10:(i+1)*10][np.argsort(c2[i*10:(i+1)*10,0])] for i in range(10)])
bm = b.reshape((10,10,2))



src =  np.array( [bm[0][0],bm[0][9],bm[9][0],bm[9][9]], np.float32)
dst = np.array( [[0,0],[450,0],[0,450],[450,450]], np.float32)
retval = cv2.getPerspectiveTransform(src,dst)
warp = cv2.warpPerspective(res2,retval,(450,450))

x=450/9
normal_color=warp.copy()
warp =cv2.cvtColor(warp ,cv2.COLOR_BGR2GRAY)
numbers=[]
thresh = cv2.threshold(gray , 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Detect horizontal lines
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,1))
# detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
# cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     cv2.drawContours(normal_color, [c], -1, (36,255,12), 3)
# cv2.imshow('img_big',normal_color)


for row in range(9):
    for col in range(9):
        
        row=row
        col=col
        
        row_top_left=row*x
        row_top_left=int(row_top_left)
        col_top_left=col*x
        col_top_left=int(col_top_left)
        
        row_bottom_right=(row+1)*x
        row_bottom_right=int(row_bottom_right)
        col_bottom_right=(col+1)*x
        col_bottom_right=int(col_bottom_right)
        
        x1,y1=row_top_left,col_top_left
        x2,y2=row_bottom_right,col_bottom_right

        cv2.rectangle(warp,(x1,y1),(x2,y2),(0,255,0),1)
        
        img_big=warp[col_top_left:col_bottom_right, row_top_left:row_bottom_right]
     
        cnts,_=cv2.findContours(img_big,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            x_small,y_small,w_small,h_small=cv2.boundingRect(cnt)
            # img=frame[y:y+h,x:x+w]
            if h_small/w_small<3:
                
                cv2.rectangle(warp,(x_small,y_small),(x_small+w_small,y_small+h_small),(0,255,0),2)
                
                img=img_big[y_small:y_small+h_small,x_small:x_small+w_small]
                
              
            # cv2.imshow('img',img)
                text=pytesseract.image_to_string(img,lang='eng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            
                if text=='\x0c':
                    numbers.append(" ")
                else:
                    numbers.append(text)


print(numbers)

cv2.imshow('warp',warp)
# plt.imshow(warp)