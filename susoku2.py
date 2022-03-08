#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def normalize_image(image_full):
    image_scaled = rescale(image_full, 0.3)
    edges = canny(image_scaled)
    #io.imshow(edges)
    selem = disk(1)
    edges = dilation(edges, selem)
    #io.imshow(edges)

    edges = (edges).astype(np.uint8)
        
    ext_contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    #fig, ax = plt.subplots()
    #ax.imshow(edges, cmap=plt.cm.gray)
        
    ext_contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    #fig, ax = plt.subplots()
    #ax.imshow(edges, cmap=plt.cm.gray)
    
    for n, contour in enumerate(ext_contours):
        contour = np.array(contour).squeeze() # we need to remove one dim, se below
        if contour.ndim > 1:
            xx=1
    #        ax.plot(contour[:, 0], contour[:, 1], linewidth=2)

    contour = max(ext_contours, key=cv2.contourArea)
    #print(contour.shape)
    contour = contour.squeeze()
    #print(contour.shape)
    
    #fig, ax = plt.subplots()
    #ax.imshow(image_scaled, cmap=plt.cm.gray)
    #ax.plot(contour[:, 0], contour[:,  1], 'r')
    
    epsilon = 0.05 * cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True).squeeze()
    
    #fig, ax = plt.subplots()
    #ax.imshow(image_scaled, cmap=plt.cm.gray)
    #ax.plot(corners[:, 0], corners[:, 1], '*r')
    
    sides=np.array([[0,0],[0,600],[600,600],[600,0]])
    sides1=np.array([[0,0],[0,252],[252,252],[252,0]])
        
    true_c=np.array([[0,0],[0,0],[0,0],[0,0]])
    for i in range(4):
        dist=[]
        for j in range(4):
            dist.append( ((corners[i][0]-sides[j][0])**2+(corners[i][1]-sides[j][1])**2)**(0.5) )
            #print(dist)
        true_c[i]=sides1[np.argmin(dist)]
    
    #true_c
    
    tform = ProjectiveTransform()
    tform.estimate(true_c, corners)
    image_warped = warp(image_scaled, tform)
    
    #fig, ax=plt.subplots()
    #ax.imshow(image_warped[:252, :252], cmap=plt.cm.gray)
    return(image_warped[:252, :252])


def recognize_digits(image_warped): 
    from skimage.feature import match_template
    numb=[]
    for i in range(9):
        for j in range(9):
            numb.append(image_warped[i*28:(i+1)*28, j*28:28*(j+1)])
        
    tmpl=[]
    for i in range(0, 10):
        filename = "%d.jpg" % i
        image_full = io.imread(filename, as_gray=True, plugin='matplotlib')
        tmpl.append(image_full)
    
    
    for i in range(0, 10):
        filename = "%d" % i + "%d.jpg" % i
        image_full = io.imread(filename, as_gray=True, plugin='matplotlib')
        tmpl.append(image_full)
    
    
    
    new_tmpl1=[1-tmp>0.53 for tmp in tmpl]
    new_tmpl2=[1-tmp>0.555 for tmp in tmpl]
    new_tmpl3=[1-tmp>0.6 for tmp in tmpl]
    new_tmpl=new_tmpl1+new_tmpl2+new_tmpl3
    
    new_num=[1-tmp>0.6 for tmp in numb]

    arr=[['.' for i in range(9)] for j in range(9)]
    for i in range(len(numb)):
        result=[]
        for shift_0 in range(-2,3):
            for shift_1 in range(-2,3):
                for j in range(len(tmpl)):
                    number=new_num[i][4:24,4:24]
                    rolled = np.roll(number,axis=0,shift=shift_0)
                    rolled = np.roll(rolled,axis=1,shift=shift_1)
                    result.append(match_template(rolled, new_tmpl[j][4:24,4:24]))
                
        #print(result)
        k=np.argmax(result)
        arr[int(i//9)][int(i%9)]=k%10
        
          
    return arr
    #rcParams['figure.figsize'] = 10, 10
    #fig, ax=plt.subplots()
    #ax.imshow(image_warped[:252, :252], cmap=plt.cm.gray)
    

