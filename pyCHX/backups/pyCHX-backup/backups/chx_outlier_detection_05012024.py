def is_outlier(points,thresh=3.5,verbose=False):
    """MAD test
    """     
    points.tolist()
    if len(points) ==1:
        points=points[:,None]
        if verbose:
            print('input to is_outlier is a single point...')
    median = np.median(points)*np.ones(np.shape(points))#, axis=0)
    
    diff = (points-median)**2
    diff=np.sqrt(diff)
    med_abs_deviation= np.median(diff)
    modified_z_score = .6745*diff/med_abs_deviation
    return modified_z_score > thresh

def outlier_mask(avg_img,mask,roi_mask,outlier_threshold = 7.5,maximum_outlier_fraction = .1,verbose=False,plot=False):
    """
    outlier_mask(avg_img,mask,roi_mask,outlier_threshold = 7.5,maximum_outlier_fraction = .1,verbose=False,plot=False)
    avg_img: average image data (2D)
    mask: 2D array, same size as avg_img with pixels that are already masked
    roi_mask: 2D array, same size as avg_img, ROI labels 'encoded' as mask values (i.e. all pixels belonging to ROI 5 have the value 5)
    outlier_threshold: threshold for MAD test
    maximum_outlier_fraction: maximum fraction of pixels in an ROI that can be classifed as outliers. If the detected fraction is higher, no outliers will be masked for that ROI.
    verbose: 'True' enables message output
    plot: 'True' enables visualization of outliers
    returns: mask (dtype=float): 0 for pixels that have been classified as outliers, 1 else
    dependency: is_outlier()

    function does outlier detection for each ROI separately based on pixel intensity in avg_img*mask and ROI specified by roi_mask, using the median-absolute-deviation (MAD) method

    by LW 06/21/2023
    """
    hhmask = np.ones(np.shape(roi_mask))
    pc=1

    for rn in np.arange(1,np.max(roi_mask)+1,1):
        rm=np.zeros(np.shape(roi_mask));rm=rm-1;rm[np.where( roi_mask == rn)]=1
        pixel = roi.roi_pixel_values(avg_img*rm, roi_mask, [rn] )
        out_l = is_outlier((avg_img*mask*rm)[rm>-1], thresh=outlier_threshold)
        if np.nanmax(out_l)>0: # Did detect at least one outlier
            ave_roi_int = np.nanmean((pixel[0][0])[out_l<1])
            if verbose: print('ROI #%s\naverage ROI intensity: %s'%(rn,ave_roi_int))
            try:
                upper_outlier_threshold = np.nanmin((out_l*pixel[0][0])[out_l*pixel[0][0]>ave_roi_int])
                if verbose: print('upper outlier threshold: %s'%upper_outlier_threshold)
            except:
                upper_outlier_threshold = False
                if verbose: print('no upper outlier threshold found')
            ind1 = (out_l*pixel[0][0])>0; ind2 =  (out_l*pixel[0][0])< ave_roi_int
            try:
                lower_outlier_threshold = np.nanmax((out_l*pixel[0][0])[ind1*ind2])
            except:
                lower_outlier_threshold = False
                if verbose: print('no lower outlier threshold found')
        else:
            if verbose:  print('ROI #%s: no outliers detected'%rn)

        ### MAKE SURE we don't REMOVE more than x percent of the pixels in the roi 
        outlier_fraction =  np.sum(out_l)/len(pixel[0][0])
        if verbose: print('fraction of pixel values detected as outliers: %s'%np.round(outlier_fraction,2))
        if outlier_fraction > maximum_outlier_fraction:
            if verbose: print('fraction of pixel values detected as outliers > than maximum fraction %s allowed -> NOT masking outliers...check threshold for MAD and maximum fraction of outliers allowed'%maximum_outlier_fraction)
            upper_outlier_threshold = False; lower_outlier_threshold = False

        if upper_outlier_threshold:
            hhmask[avg_img*rm > upper_outlier_threshold] = 0
        if lower_outlier_threshold:
            hhmask[avg_img*rm < lower_outlier_threshold] = 0

        if plot:
            if pc == 1: fig,ax = plt.subplots(1,5,figsize=(24,4))
            plt.subplot(1,5,pc);pc+=1;
            if pc>5: pc=1
            pixel = roi.roi_pixel_values(avg_img*rm*mask, roi_mask, [rn] )
            plt.plot( pixel[0][0] ,'bo',markersize=1.5 )
            if upper_outlier_threshold or lower_outlier_threshold:
                x=np.arange(len(out_l))
                plt.plot([x[0],x[-1]],[ave_roi_int,ave_roi_int],'g--',label='ROI average: %s'%np.round(ave_roi_int,4))
            if upper_outlier_threshold:
                ind=(out_l*pixel[0][0])> upper_outlier_threshold
                plt.plot(x[ind],(out_l*pixel[0][0])[ind],'r+')
                plt.plot([x[0],x[-1]],[upper_outlier_threshold,upper_outlier_threshold],'r--',label='upper thresh.: %s'%np.round(upper_outlier_threshold,4))
            if lower_outlier_threshold:
                ind=(out_l*pixel[0][0])< lower_outlier_threshold
                plt.plot(x[ind],(out_l*pixel[0][0])[ind],'r+')
                plt.plot([x[0],x[-1]],[lower_outlier_threshold,lower_outlier_threshold],'r--',label='lower thresh.: %s'%np.round(upper_outlier_threshold,4))
            plt.ylabel('Intensity') ;plt.xlabel('pixel');plt.title('ROI #: %s'%rn);plt.legend(loc='best',fontsize=8)

    if plot:
        fig,ax = plt.subplots()
        plt.imshow(hhmask)
        hot_dark=np.nonzero(hhmask<1)
        cmap = plt.cm.get_cmap('viridis')
        plt.plot(hot_dark[1],hot_dark[0],'+',color=cmap(0))
        plt.xlabel('pixel');plt.ylabel('pixel');plt.title('masked pixels with outlier threshold: %s'%outlier_threshold)

    return hhmask
