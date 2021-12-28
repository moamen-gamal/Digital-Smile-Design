import numpy as np
import cv2
import matplotlib.pyplot as plt


alpha = 0.4
img = cv2.imread('./images/face3.jpg')

img_blur = cv2.GaussianBlur(img, (7,7), 0) 
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=120) 
# cv2.imshow('Canny Edge Detection', edges)
cv2.imwrite('result_edge.jpg', edges)

img2 = cv2.imread('result_edge.jpg')

#(height, width) = img.shape[:2]
#img2 = cv2.resize(img2, (int(width), int(height )), interpolation = cv2.INTER_CUBIC)

out_img = np.zeros(img.shape,dtype=img.dtype)
out_img[:,:,:] = (alpha * img[:,:,:]) + ((1-alpha) * img2[:,:,:])
# cv2.imshow('Output blend',out_img)
cv2.imwrite('result_blend.jpg', out_img)
cv2.waitKey(0)

# TARGET = (439,90)
Horizontal = 500
Vertical = 120

imgD = edges
def findNearestWhite(imgD, horizontal, vertical ):
    
    nonzero = np.argwhere(imgD == 255) #white & vertical 
    Hline=  nonzero[nonzero[:,1] == horizontal] #lay on the horizontal line y=const
    distances = np.array(Hline[:,0] - vertical )  # nearest point to the line 
    nearest_index = np.argmin(distances)
    if(distances[nearest_index] == 0) :print("medline is perfect")
    else: print("medline is shifted with ", distances[nearest_index])
    plt.imshow(imgD)
    plt.axvline(x=Hline[nearest_index][0], ymin=0.05, ymax=0.95, color='green', label='axvline - % of full height')

    plt.show()

    return Hline[nearest_index]



print (findNearestWhite(imgD, Horizontal, Vertical)) 



cv2.destroyAllWindows()

