# Gradient in x-direction
def Sobelx(image):
    sobelx = np.array([[1,0,-1],
                    [2,0,-2],
                    [1,0,-1]])
    Ix = nd.convolve(image,sobelx)           
    return Ix


# Gradient in y-direction
def Sobely(image):
    sobely = np.array([[ 1, 2, 1],
                    [ 0, 0, 0],
                    [-1,-2,-1]])

    Iy = nd.convolve(image,sobely)
    return Iy


# Convert RGB to grayscale
def rgb2gray(image):
    image = np.array(image)
    r, c, ch = image.shape
    gray = np.zeros([r,c])

    gray = np.mean(image, axis=2)
    return gray


# Reduce image size
def reduceRes(image,factor):
    r,c = image.size
    r = np.int(np.floor(r/factor))
    c = np.int(np.floor(c/factor))
    new = image.resize((r,c))
    return new


# Box filter for smoothing
def boxFilter(image,rc):
    box = np.ones([rc,rc])/rc**2
    image = nd.convolve(gray,box)
    return image


# Pad image with zeros
def pad(image,filter):
    r,c = filter.shape
    factor = int(np.floor(r/2))

    r_img, c_img = image.shape
    pad = np.zeros([r_img,factor])
    image = np.hstack([pad,image,pad])

    r_img, c_img = image.shape
    pad = np.zeros([factor,c_img])
    image = np.vstack([pad,image,pad])

    return image


# Peak finding for nonmax supression
def peaks(image,window):
    factor = int(np.floor(window.shape[0]/2))
    image = pad(image,window)
    peaks = np.zeros([image.shape[0],image.shape[1]])

    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            window = image[i-1:i+2,j-1:j+2]
            if image[i,j]==np.amax(window):
                peaks[i,j] = image[i,j]


    # remove padding
    peaks = peaks[factor:-factor,factor:-factor]

    return peaks


def norm_cross_corr(image,filter):
    r,c = image.shape
    ncc = np.zeros([r,c])
    
    padded = pad(image,filter)

    for i in range(r):
        for j in range(c):
            f = padded[i:i+filter.shape[0],j:j+filter.shape[1]]
            len_f = np.sqrt(np.sum(f**2))
            if len_f==0:
                norm_f = np.zeros(f.shape)
            else:
                norm_f = f/len_f
            norm_g = filter/np.sqrt(np.sum(filter**2))

            if filter.size != norm_f.size:
                print('Norm F:', norm_f)
                print('Filter:',filter)
            
            norm_f = np.reshape(norm_f,[1,filter.size])
            norm_g = np.reshape(norm_g,[filter.size,1])
            ncc[i,j] = np.dot(norm_f,norm_g)
            
    return ncc


# Harris Corner Detector
def HarrisCorner(image,threshold):

    # Find gradients
    Ix = Sobelx(image)
    Iy = Sobely(image)
    Ixx = Ix**2
    Iyy = Iy**2

    # Compute R values
    detM = Ixx*Iyy
    traceM = Ixx+Iyy
    k = 0.05
    R = detM - k*traceM**2

    # Threshold values
    R[R <= threshold*np.amax(R)] = 0
    
    # Nonmax Supression
    R = peaks(R,np.ones([3,3]))

    return R


# Reduce threshold until ~1% of pixels are corners
def reducedHarris(image,ratio):
    threshold = 1.0
    while ratio >= 100:
        threshold = 0.1*threshold

        corners = HarrisCorner(image,threshold)
        # Number of corners
        idx = np.where(corners>0)
        idx = np.array(idx).T
        ratio = image.size/idx.shape[0]

    print('Threshold:',threshold)
    print('Corners:',idx.shape[0])

    return corners


# Improve corners from HCD
def neighboringCorner(corners,width,num_corners):
    # Remove isolated corners in a window

    good_corners = np.copy(corners)
    good_corners[good_corners>0] = 1
    good_corners[good_corners<0] = 0
    print('Previous corners:', np.sum(good_corners))

    filter = np.ones([width,width])
    neighbors = nd.convolve(good_corners,filter)

    neighbors[neighbors<num_corners] = 0
    neighbors[neighbors>=num_corners] = 1
    neighbors = neighbors*good_corners

    print('Better corners:',np.sum(neighbors))

    return corners*neighbors


# Find matches from im1 to im2
def findMatches(corners1,corners2,width,num_matches):
    
    matches = np.zeros([num_matches,4])
    i = 0
    while i<num_matches:
        # Get a random patch location
        idx = np.where(corners1>0)
        idx = np.array(idx).T
        rand_corner = np.random.randint(idx.shape[0],size=1)
        loc = idx[rand_corner,:]
        loc = np.concatenate(loc)
        r = loc[0]
        c = loc[1]

        # Patch from location
        left = np.int( (width-1)/2 )
        right = np.int( (width-1)/2 + 1 )
        patch = np.around(corners[r-left:r+right,c-left:c+right])

        # Match the patch to second image
        fit = norm_cross_corr(corners2,patch)
        best = np.where(fit==np.amax(fit))
        best = np.array(best).T

        if best.shape[0] == 1:
            print("Match #",i+1)
            print(np.round(np.amax(fit),3))
            best = np.concatenate(best)
            matches[i,:] = np.array([r,c,best[0],best[1]])
            i+=1

    return matches


def drawLines(image1,image2,matches):

    offset = image1.shape[0]
    x = np.copy(matches[:,0:4:2])
    x[:,1] = x[:,1] + offset
    y = np.copy(matches[:,1:4:2])

    both = np.vstack([image1,image2])
    plt.imshow(both,cmap='gray')
    for i in range(0,x.shape[0]):
        plt.plot(y[i,:],x[i,:])
    
    return

