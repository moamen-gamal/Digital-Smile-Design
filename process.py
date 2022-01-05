import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os, io

from PIL import Image, ImageDraw
import cv2
import numpy as np
from numpy.core.fromnumeric import shape, size
import matplotlib as plt

global Zefer
Zefer = 0
global width21
width21 =0


def template(fname,tempfilename):
    img = cv2.imread(fname)
    face = rect = Image.open(fname)


    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
    
    xid =[]
    yid =[]
    pts = np.empty([1,1])
    pts.fill(0)
    ptss =[]
    i =0
    #13 mid upper lip
    ids = [80,89, 310,319]

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            for id,lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                xid.append(x)
                yid.append(y)

    for i in range(0,len(ids)):
        ptss.append([xid[ids[i]],yid[ids[i]]])
    ptsss = np.array(ptss)
    
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(ptsss)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    cv2.imwrite('dst2-temp.jpg', croped)


    template =Image.open(tempfilename)
    rectangle = Image.open('./dst2-temp.jpg')


    resizedTeeth = template.convert("RGBA").resize(rectangle.size)
    face.paste(resizedTeeth, (ptss[0]), mask = resizedTeeth)
    face.save('./faceTemp.png')
    img = Image.open(r'faceTemp.png')
    return img,rectangle.size


def ApplyColoration(fname,rangesid):

    ranges =[(234 ,223 ,195),(255 ,255 ,255),(231,221,197),(228,211,169)]
    range_select = ranges[rangesid]
    img = cv2.imread(fname)
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    xid =[]
    yid =[]
    pts = np.empty([1,1])
    pts.fill(0)
    ptss =[]
    i =0
    ids = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            #mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
            #                      drawSpec,drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                #print(lm)
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                xid.append(x)
                yid.append(y)
                #print(id,x,y)
    #print(xid,yid)
    for i in range(0,len(ids)):
        ptss.append([xid[ids[i]],yid[ids[i]]])
    
    ptsss = np.array(ptss)
    
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(ptsss)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    ptsss = ptsss - ptsss.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [ptsss], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    imgE = dst
    sensitivity = 90
    lower = np.array([8,0,255-sensitivity])
    upper = np.array([172,sensitivity,255])


    # turn image into hsv
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    # mask that makes any non white black
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(dst,dst, mask= mask)
    cv2.imwrite("output.jpg", output)
    cv2.imwrite('dst.jpg',dst)
    capOpener = Image.open(r"output.jpg")
    img_blur = cv2.GaussianBlur(imgE, (5,5), 0)
    edges = cv2.Canny(image=img_blur, threshold1=10, threshold2=120)
    cv2.imwrite('edges.jpg',edges)
    #width33, height33 = capOpener221.size

    # Get the size of the image
    width21, height21 = capOpener.size
    bWANTEDArray=[]
    jArray=[]

    rsummer=0
    gsummer=0
    bsummer=0
    N_points=0
    for x in range(0,width21):
        for y in range (0,height21):
            current_color = capOpener.getpixel( (x,y) )
            #print(current_color)
            r,g,b= current_color
            if(r>0):
                rsummer+=r
                gsummer+=g
                bsummer+=b
                N_points+=1


    for x in range(0,width21):
        for y in range (0,height21):
            current_color = capOpener.getpixel( (x,y) )
            #print(current_color)

            r,g,b= current_color
            if(b > 0 and r > 0 and g > 0  ):
                bWANTEDArray.append(x)
                jArray.append(y)
            else:
                capOpener.putpixel((x, y), (0, 0, 0))

    
    for x, y in zip(bWANTEDArray, jArray):
            
        capOpener.putpixel((x, y), range_select)

    
    THEMODIFIED=Image.open(fname)
    for x in range(xid[78],xid[308]):
        for y in range (yid[13],yid[14]):
            current_color = capOpener.getpixel((x-xid[78], y-yid[13]))
            #print(current_color)

            r, g, b = current_color


            if(r>0 and g>0 and b>0):
                THEMODIFIED.putpixel((x, y),current_color)

    return THEMODIFIED
    
    
def gumDetection(fname):
    img = cv2.imread(fname)
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    xid =[]
    yid =[]
    pts = np.empty([1,1])
    pts.fill(0)
    ptss =[]
    i =0
    ids = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            #mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
            #                      drawSpec,drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                #print(lm)
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                xid.append(x)
                yid.append(y)
                #print(id,x,y)
    #print(xid,yid)
    for i in range(0,len(ids)):
        ptss.append([xid[ids[i]],yid[ids[i]]])
    
    ptsss = np.array(ptss)
    
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(ptsss)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    cv2.imwrite("cropped.jpg",croped)
    ## (2) make mask
    ptsss = ptsss - ptsss.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [ptsss], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    imgE = dst
    
    cv2.imwrite('dst.jpg',dst)
    img_blur = cv2.GaussianBlur(imgE, (5,5), 0)
    mean, std = cv2.meanStdDev(croped)
    TH1 =int(mean[0]-std[0])
    TH2 = int(mean[0]+std[0])
    edges = cv2.Canny(image=img_blur, threshold1=TH1, threshold2=TH2)
    cv2.imwrite("edges.jpg",edges)
    Xstorer =[]
    capOpener221  = Image.open(r"edges.jpg")
    width33, height33 = capOpener221.size
    R=0
    for x in range(0,width33):
        for y in range (0,height33):
            if(capOpener221.getpixel((x,y))>=250):
                R=R+1
                Xstorer.append(y)

    MaxStorer=max(Xstorer)
    #print(Xstorer)
    Zaree=0
    Final22=0
    AverageArr=[]
    for i in range(0,len(Xstorer)):
        Zaree=MaxStorer-Xstorer[i]
        if(Zaree>=5 and Zaree<=10):
            Final22=Zaree
            AverageArr.append(Final22)

    Averagenumber=sum(AverageArr)/len(AverageArr)
    #print(Averagenumber)
    if(Averagenumber>7.7 and Averagenumber<8.2):
        return 0
    else:
        return 1

def MidlineDrawing(fname):
    img = cv2.imread(fname)
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

    xid =[]
    yid =[]
    pts = np.empty([1,1])
    pts.fill(0)
    ptss =[]
    i =0
    #13 mid upper lip
    ids = [8,200,78,13]
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            for id,lm in enumerate(faceLms.landmark):
                
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                xid.append(x)
                yid.append(y)
                
    for i in range(0,len(ids)):
        ptss.append([xid[ids[i]],yid[ids[i]]])
    img2 = Image.open(fname)
    draw = ImageDraw.Draw(img2) 
    draw.line((ptss[0][0],ptss[0][1],ptss[1][0],ptss[1][1]), fill=128)

    r = gumDetection(fname)
    ########

    image = Image.open(r"cropped.jpg")
    width21,height21 = image.size
    Classify_Number=0
    ZeroNumver=0
    Neon=0
    NeonArray=[]
    for i in range (1,height21):
        Neon=0
        for j in range(1,width21):
            current_color = image.getpixel((j,i))
            r,g,b= current_color
            b = int(b)
            g = int(g)
            r = int(r)
            if(b<=40 and r <40 and g<40):
                ZeroNumver=ZeroNumver+1
            if(b>=180 and r>=180 and g>=180 ):
                Classify_Number=Classify_Number+1
                Neon=Neon+1
        NeonArray.append(Neon)
    for i in range(np.size(NeonArray)):
        if(NeonArray[i]==max(NeonArray)):
            Zefer=i
    ######
    croppedimg = cv2.imread("cropped.jpg")
    img_blur = cv2.GaussianBlur(croppedimg, (5,5), 0)
    mean, std = cv2.meanStdDev(croppedimg)
    TH1 =int(100)
    TH2 = int(200)
    edges2 = cv2.Canny(image=img_blur, threshold1=TH1, threshold2=TH2)
    Horizontal = Zefer
    Vertical = width21/2
    cv2.imwrite("edges2.jpg",edges2)
    
    Hline = findNearestWhite(edges2, Horizontal, Vertical)
 
    center = Hline[0]
    draw = ImageDraw.Draw(img2)
    wid,heig = image.size
    draw.line((ptss[2][0]+1.5*center,0, ptss[2][0]+1.5*center,100*height21), fill=(0,255,0))
    return img2
             

def colorationDetection(fname):
    cap = cv2.imread(r"dst.jpg")
    width21, height21 = Image.open(r"dst.jpg").size

    Classify_Number=0
    ZeroNumver=0
    Neon=0
    NeonArray=[]
    for i in range (1,height21):
        Neon=0
        for j in range(1,width21):
            b,g,r = cap[i,j]
            b = int(b)
            g = int(g)
            r = int(r)
            if(b<=40 and r <40 and g<40):
                ZeroNumver=ZeroNumver+1
            if(b>=180 and r>=180 and g>=180 ):
                Classify_Number=Classify_Number+1
                Neon=Neon+1
        NeonArray.append(Neon)
    for i in range(np.size(NeonArray)):
        if(NeonArray[i]==max(NeonArray)):
            Zefer=i

    Definer=Classify_Number/((width21*height21)-ZeroNumver)

    return Definer


def gab_Detection(fname):
    img = cv2.imread(fname)
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

    xid =[]
    yid =[]
    pts = np.empty([1,1])
    pts.fill(0)
    ptss =[]
    i =0
    #13 mid upper lip
    ids = [82,13,312,317,14,87]
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            for id,lm in enumerate(faceLms.landmark):
                
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                xid.append(x)
                yid.append(y)
                
    for i in range(0,len(ids)):
        ptss.append([xid[ids[i]],yid[ids[i]]])
    ptsss = np.array(ptss)
    
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(ptsss)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    ptsss = ptsss - ptsss.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [ptsss], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    mean, std = cv2.meanStdDev(dst)

    image = dst
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set threshold level
    threshold_level = mean[0] - std[0]

    # Find coordinates of all pixels below threshold
    coords = np.column_stack(np.where(gray < threshold_level))
    #print(coords)

    # Create mask of all pixels lower than threshold level
    mask = gray < threshold_level

    # Color the pixels in the mask
    image[mask] = (204, 119, 0)
    black =0
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            if image[i][j][0] == 204 and image[i][j][1] == 119 and image[i][j][2] == 0  :
                black =black + 1

    pixTotal =image.shape[0]* image.shape[1]

    if(black/ pixTotal > 0.05):
        return 1
    else:
        return  0   
   

def findNearestWhite(edges, horizontal, vertical):
    nonzero = np.argwhere(edges == 255) #white & vertical
    width = vertical*2 
    Hline1 =  nonzero[nonzero[:, 1] >= width/5]
    Hline2 = Hline1[Hline1[:, 1] <= width/2] #lay on the horizontal line y=const

    distances = np.array(abs(Hline2[:,0] - vertical) )  # nearest point to the line 
    nearest_index = np.argmin(distances)
    # plt.imshow(edges)
    # plt.axvline(x=Hline2[nearest_index][0], ymin=0.05, ymax=0.95, color='green', label='axvline - % of full height')

    # plt.show()

    return Hline2[nearest_index]