# import libraries here
import numpy as np
import cv2 # OpenCV biblioteka

def count_rbc_for_large_peak(image):
    grayImage= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rgbImage = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    equalizedImage= cv2.equalizeHist(grayImage)


    mask_gray = cv2.adaptiveThreshold(equalizedImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,135,0)

    kernel = np.ones((3,3),np.uint8)
    opening= cv2.morphologyEx(mask_gray,cv2.MORPH_OPEN,kernel,iterations=3)

    sure_bg= cv2.dilate(opening,kernel,iterations=3)

    dist_transform= cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret,sure_fg= cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

    sure_fg= np.uint8(sure_fg)

    img, contours, hierarchy = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    contoursOfRBC= [] 
    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        width,height= size
        if width>6 or height >6:
            contoursOfRBC.append(contour)

    return len(contoursOfRBC)

def count_rbc(image):
    grayImage= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rgbImage = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    equalizedImage= cv2.equalizeHist(grayImage)

    hist_full = cv2.calcHist([rgbImage], [0], None, [255],[0,255])

    hist_full = [val[0] for val in hist_full]
    indices= list(range(0,256))
    s=[(x,y) for y,x in sorted(zip(hist_full,indices), reverse=True)]

    index_of_highest_peak=s[0][0]

    mask_gray = cv2.adaptiveThreshold(equalizedImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,205,0)

    kernel = np.ones((3,3),np.uint8)
    opening= cv2.morphologyEx(mask_gray,cv2.MORPH_OPEN,kernel,iterations=3)

    sure_bg= cv2.dilate(opening,kernel,iterations=3)

    dist_transform= cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret,sure_fg= cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)

    sure_fg= np.uint8(sure_fg)

    img, contours, hierarchy = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    contoursOfRBC= [] 
    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        width,height= size
        if width>7 or height >7:
            contoursOfRBC.append(contour)

    return len(contoursOfRBC)

def count_wbc(image): 
    hsv= cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    
    lower_gray = np.array([22,100,105], np.uint8)
    upper_gray = np.array([177,165,255], np.uint8)
    mask_gray= cv2.inRange(hsv,lower_gray,upper_gray)


    im_floodfill = mask_gray.copy()

    h,w= mask_gray.shape[:2]
    mask1= np.zeros((h+2,w+2),np.uint8)

    cv2.floodFill(im_floodfill,mask1,(0,0),255)

    im_floodfill_inv= cv2.bitwise_not(im_floodfill)

    im_out= mask_gray | im_floodfill_inv

    kernel = np.ones((2,2),np.uint8)
    opening= cv2.morphologyEx(im_out,cv2.MORPH_OPEN,kernel,iterations=1)

    sure_bg= cv2.dilate(opening,kernel,iterations=3)

    dist_transform= cv2.distanceTransform(opening,cv2.DIST_L1,5)
    ret,sure_fg= cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

    sure_fg= np.uint8(sure_fg)

    img, contours, hierarchy = cv2.findContours(sure_fg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    contoursOfWBC= [] 
    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        width,height= size
        if width>19 and height >19:
            contoursOfWBC.append(contour)

    return len(contoursOfWBC)

def count_histogram_peak(image):
    calculatedHist = cv2.calcHist([image], [0], None, [255], [0, 255])
 
    calculatedHist = [val[0] for val in calculatedHist]
    indices= list(range(0,256))
    sort= [(x,y) for y,x in sorted(zip(calculatedHist,indices), reverse=True)]

    index_of_highest_peak = sort[0][0]
    return index_of_highest_peak

def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj crvenih krvnih zrnaca, belih krvnih zrnaca i
    informaciju da li pacijent ima leukemiju ili ne, na osnovu odnosa broja krvnih zrnaca

    Ova procedura se poziva automatski iz main procedure i taj deo kod nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih crvenih krvnih zrnaca,
             <int> broj prebrojanih belih krvnih zrnaca,
             <bool> da li pacijent ima leukemniju (True ili False)
    """
    red_blood_cell_count = 0
    white_blood_cell_count = 0


    image= cv2.imread(image_path) # read image from file

    highestPeak= count_histogram_peak(image)
    hsvImage= cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    if(highestPeak<245):
        averageHist= np.average(hsvImage)

        if(averageHist>106):
            white_blood_cell_count = count_wbc(image)
            red_blood_cell_count = count_rbc(image)
        else:
            white_blood_cell_count = count_wbc(image) 
            red_blood_cell_count = count_rbc(image)-white_blood_cell_count
    else:
            white_blood_cell_count = count_wbc(image)
            red_blood_cell_count = count_rbc_for_large_peak(image)

    print(image_path)
    print(white_blood_cell_count)
    print(red_blood_cell_count)

    wbcValue= white_blood_cell_count

    if(wbcValue>=5):
        wbcValue=wbcValue*1.3

    if wbcValue/red_blood_cell_count>0.09: 
        has_leukemia=True
    else:
        has_leukemia=False

    print(wbcValue/red_blood_cell_count)
    print(has_leukemia)
    # TODO - Prebrojati crvena i bela krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure

    # TODO - Odrediti da li na osnovu broja krvnih zrnaca pacijent ima leukemiju i vratiti True/False kao povratnu vrednost ove procedure

    return red_blood_cell_count, white_blood_cell_count, has_leukemia
