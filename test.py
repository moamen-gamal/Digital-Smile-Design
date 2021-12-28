import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
img = Image.open('C:\\Users\\moamen\\Desktop\\Digital-Smile-Design\\Images\\02.jpg')
img = cv2.imread('C:\\Users\\moamen\\Desktop\\Digital-Smile-Design\\Images\\02.jpg')
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
ids = [10,19,0,152]
ids2 =[10,152]
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
            # print(id,x,y)
# print(xid,yid)
for i in range(0,len(ids)):
    ptss.append([xid[ids[i]],yid[ids[i]]])
ptss2=[]
for i in range(0,len(ids2)):
    ptss2.append([xid[ids2[i]],yid[ids2[i]]])
#print(ptss)
#print(ptss[0])
img = Image.open('C:\\Users\\moamen\\Desktop\\Digital-Smile-Design\\Images\\02.jpg')      
draw = ImageDraw.Draw(img) 
draw.line((ptss[0][0],ptss[0][1],ptss[1][0],ptss[1][1],ptss[2][0],ptss[2][1],ptss[3][0],ptss[3][1]), fill=128)

draw = ImageDraw.Draw(img) 
draw.line((ptss2[0][0],ptss2[0][1],ptss2[1][0],ptss2[1][1]), fill=(0,255,0))
img.save("ddd.jpg")
#print(img)
