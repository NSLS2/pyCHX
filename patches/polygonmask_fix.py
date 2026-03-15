# fix for newer environments on jupyter hub: polygon( y,x, shape = image.shape) -> need to be specific about shape
# Not needed for older, local environments on srv1,2,3. Fix above will probably also work there, BUT there would be a problem with disk <-> circle that has been renamed in skimage at some point
# -> older, local environments on srv1,2,3 work without this fix, only apply when running on jupyter hub
def create_multi_rotated_rectangle_mask(  image, center=None, length=100, width=50, angles=[0] ):
    ''' Developed at July 10, 2017 by Y.G.@CHX, NSLS2
     Create multi rectangle-shaped mask by rotating a rectangle with a list of angles
     The original rectangle is defined by four corners, i.e.,
         [   (center[1] - width//2, center[0]),
             (center[1] + width//2, center[0]),
             (center[1] + width//2, center[0] + length),
             (center[1] - width//2, center[0] + length)
          ]

     Parameters:
         image: 2D numpy array, to give mask shape
         center: integer list, if None, will be the center of the image
         length: integer, the length of the non-ratoted rectangle
         width: integer,  the width of the non-ratoted rectangle
         angles: integer list, a list of rotated angles

    Return:
        mask: 2D bool-type numpy array
    '''

    from skimage.draw import  polygon
    from skimage.transform import rotate
    cx,cy = center
    imy, imx = image.shape
    mask = np.zeros( image.shape, dtype = bool)
    wy =  length
    wx =   width
    x = np.array( [ max(0, cx - wx//2), min(imx, cx+wx//2), min(imx, cx+wx//2), max(0,cx-wx//2 ) ])
    y = np.array( [ cy, cy, min( imy, cy + wy) , min(imy, cy + wy) ])
    rr, cc = polygon( y,x, shape = image.shape)
    mask[rr,cc] =1
    mask_rot=  np.zeros( image.shape, dtype = bool)
    for angle in angles:
        mask_rot +=  np.array( rotate( mask, angle, center= center ), dtype=bool) #, preserve_range=True)
    return  ~mask_rot

def create_cross_mask(  image, center, wy_left=4, wy_right=4, wx_up=4, wx_down=4,
                     center_disk = True, center_radius=10
                     ):
    '''
    Give image and the beam center to create a cross-shaped mask
    wy_left: the width of left h-line
    wy_right: the width of rigth h-line
    wx_up: the width of up v-line
    wx_down: the width of down v-line
    center_disk: if True, create a disk with center and center_radius

    Return:
    the cross mask
    '''
    from skimage.draw import line_aa, line, polygon, disk

    imy, imx = image.shape
    cx,cy = center
    bst_mask = np.zeros_like( image , dtype = bool)
    ###
    #for right part
    wy = wy_right
    x = np.array( [ cx, imx, imx, cx  ])
    y = np.array( [ cy-wy, cy-wy, cy + wy, cy + wy])
    rr, cc = polygon( y,x,shape=image.shape)
    bst_mask[rr,cc] =1

    ###
    #for left part
    wy = wy_left
    x = np.array( [0,  cx, cx,0  ])
    y = np.array( [ cy-wy, cy-wy, cy + wy, cy + wy])
    rr, cc = polygon( y,x,shape=image.shape)
    bst_mask[rr,cc] =1

    ###
    #for up part
    wx = wx_up
    x = np.array( [ cx-wx, cx + wx, cx+wx, cx-wx  ])
    y = np.array( [ cy, cy, imy, imy])
    rr, cc = polygon( y,x,shape=image.shape)
    bst_mask[rr,cc] =1

    ###
    #for low part
    wx = wx_down
    x = np.array( [ cx-wx, cx + wx, cx+wx, cx-wx  ])
    y = np.array( [ 0,0, cy, cy])
    rr, cc = polygon( y,x,shape=image.shape)
    bst_mask[rr,cc] =1

    if center_radius!=0:
        rr, cc = disk((cy, cx), center_radius, shape = bst_mask.shape)
        bst_mask[rr,cc] =1


    full_mask= ~bst_mask

    return full_mask