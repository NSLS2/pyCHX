from pyCHX.chx_packages import *
import sys

def dispersion(dispersion_type,x,*args):
    """
    general dispersion equation
    dispersion(dispersion_type,x,*args)
    dispersion_type: one of 'Qsquared', 'Qlinearoffset', 'Qlinear' or 'Qn'
    Qsquared:y=a*x**2
    Qlinearoffset: y=a*x+b
    Qlinear: y=a*x
    QN: y=a*x**b
    x: Q-values
    *args: a or a,b according to equations avbove
    returns: y
    """
    if dispersion_type == 'Qsquared':
        y=args[0]*x**2
    elif dispersion_type == 'Qlinearoffset':
        y=args[0]*x+args[1]
    elif dispersion_type == 'Qlinear':
         y=args[0]*x
    elif dispersion_type == 'QN':
        y=args[0]*x**args[1]
    else: raise Exception('ERROR: dispersion_type %s  is unknown!'%dispersion_type)
    return y

def g2_double(x, a, b, c, d,e,f,g):
    return  a * (b*np.exp(-(c*x)**d)+(1-b)*np.exp(-(e*x)**f))**2+g 

def g2_single(x, a, b, c, d):
    return  a * np.exp(-2*(b*x)**c)+d 

def scale_fit(x,a):
    return a*x

def power_law(x,a,b):
    return a*x**b

def log10_power_law(x,a,b):
    return np.log10(power_law(x,a,b))

def ps(x,y,shift=.5):
    from scipy.special import erf
    from scipy.optimize import curve_fit
    
    PEAK=x[np.argmax(y)]
    PEAK_y=np.max(y)
    COM=np.sum(x * y) / np.sum(y)
    
    ### from Maksim: assume this is a peak profile:
    def is_positive(num):
        return True if num > 0 else False

    # Normalize values first:
    ym = (y - np.min(y)) / (np.max(y) - np.min(y)) - shift  # roots are at Y=0

    positive = is_positive(ym[0])
    list_of_roots = []
    for i in range(len(y)):
        current_positive = is_positive(ym[i])
        if current_positive != positive:
            list_of_roots.append(x[i - 1] + (x[i] - x[i - 1]) / (abs(ym[i]) + abs(ym[i - 1])) * abs(ym[i - 1]))
            positive = not positive
    if len(list_of_roots) >= 2:
        FWHM=abs(list_of_roots[-1] - list_of_roots[0])
        CEN=list_of_roots[0]+0.5*(list_of_roots[1]-list_of_roots[0])
        profile='peak'

    else:    # ok, maybe it's a step function..
        print('no peak...trying step function...')  
        ym = ym + shift
        def err_func(x, x0, k=2, A=1,  base=0 ):     #### erf fit from Yugang
            return base - A * erf(k*(x-x0))
        popt, pcov = curve_fit(err_func, x,ym,p0=[np.mean(x),2,1.,0.])
        CEN=popt[0];FWHM=popt[1]
        profile='step'
    ps_dict={'PEAK':PEAK,'COM':COM,'FWHM':FWHM,'CEN':CEN,'profile':profile}
    return ps_dict

def skew_Lorentzian(x,a,b,c,d,e,f):
    """
    a: assymetry parameter (0 ~ symmetric)
    b: width
    c:center
    d: A
    e: slope of linear offset
    f: constant offset
    source: https://www.webpages.uidaho.edu/brauns/vibspect1.pdf
    """
    h=2*b/(1+np.exp(a*(x-c)))
    lin=e*x+f
    y=(2*d/(np.pi*h))/(1+4*((x-c)/h)**2)+lin
    return y

def pseudo_Voigt(x,a,b,c,d,e,f):
    """
    a: center
    b: amplitude
    c: FWHM
    d: mixing parameter (Gaussian/Lorentzian)
    e: constant offset
    f: slope of linear offset
    [https://www.webpages.uidaho.edu/brauns/vibspect1.pdf]
    """
    gam=0.5*c
    w=c/(2*np.sqrt(2*np.log(2)))
    eta=d
    y = b*(d/(np.pi*gam)*(gam**2/((x-a)**2+gam**2))+(1-eta)*(1/(w*np.sqrt(2*np.pi))*np.exp(-(x-a)**2/(2*w**2))))+e+f*x
    return y

def norm_pseudo_Voigt(x, a, b, c, d, e, f):# move to standard functions and import
    y = pseudo_Voigt(x, a, b, c, d, e, f)
    return y/max(y)

def double_Voigt(x,a,b,c,d,e,f,a1,b1,c1,d1):
    """
    a, a1: centers
    b, b1: amplitudes
    c, c1: FWHMs
    d, d1: mixing parameters (Gaussian/Lorentzian)
    e: constant offset
    f: slope of linear offset
    """
    gam=0.5*c;gam1=0.5*c1
    w=c/(2*np.sqrt(2*np.log(2)));w1=c1/(2*np.sqrt(2*np.log(2)))
    eta=d;eta1=d1
    y1 = b*(d/(np.pi*gam)*(gam**2/((x-a)**2+gam**2))+(1-eta)*(1/(w*np.sqrt(2*np.pi))*np.exp(-(x-a)**2/(2*w**2))))+e+f*x
    y2 = b1*(d1/(np.pi*gam1)*(gam1**2/((x-a1)**2+gam1**2))+(1-eta1)*(1/(w1*np.sqrt(2*np.pi))*np.exp(-(x-a1)**2/(2*w**2))))
    y=y1+y2
    return y

def phi_test_Voigt(x,a,b,c,d,e,f,b1,c1,d1):
    """
    function for testing azimuthal dependence
    variant of double_Voigt, with a fixed dependence (a1=a+180) between the two centers
    x in degrees
    a  centers -> a1=a+180 (deg)
    b, b1: amplitudes
    c, c1: FWHMs
    d, d1: mixing parameters (Gaussian/Lorentzian)
    e: constant offset
    f: slope of linear offset
    """
    a1=a+180
    a2=a-180
    a3=a+360
    a4=a-360
    gam=0.5*c;gam1=0.5*c1
    w=c/(2*np.sqrt(2*np.log(2)));w1=c1/(2*np.sqrt(2*np.log(2)))
    eta=d;eta1=d1
    y1 = b*(d/(np.pi*gam)*(gam**2/((x-a)**2+gam**2))+(1-eta)*(1/(w*np.sqrt(2*np.pi))*np.exp(-(x-a)**2/(2*w**2))))+e+f*x
    y2 = b1*(d1/(np.pi*gam1)*(gam1**2/((x-a1)**2+gam1**2))+(1-eta1)*(1/(w1*np.sqrt(2*np.pi))*np.exp(-(x-a1)**2/(2*w**2))))
    y3 = b1*(d1/(np.pi*gam1)*(gam1**2/((x-a2)**2+gam1**2))+(1-eta1)*(1/(w1*np.sqrt(2*np.pi))*np.exp(-(x-a2)**2/(2*w**2))))
    y4 = b1*(d1/(np.pi*gam1)*(gam1**2/((x-a3)**2+gam1**2))+(1-eta1)*(1/(w1*np.sqrt(2*np.pi))*np.exp(-(x-a3)**2/(2*w**2))))
    y5 = b1*(d1/(np.pi*gam1)*(gam1**2/((x-a4)**2+gam1**2))+(1-eta1)*(1/(w1*np.sqrt(2*np.pi))*np.exp(-(x-a4)**2/(2*w**2))))
    y=y1+y2+y3+y4+y5
    return y


################ functions related to Orientation Parameter Analysis  #################################
def I_GND(x,xc,hw,beta,rel):
    """
    generalized normal distribution: can be used as order parameter distribution to extract order parameters from SAXS intensities
    I_GND(x,xc,hw,beta)
    x: angle [deg]
    xc: list of center position(s) of peak(s) [deg], NOTE: will automatically generate twin-peak for pi-symmetry
    hw: list of half width(s) of peak(s) [deg]
    beta: sharpness parameter: 2: wide, 1: sharp
    rel: list of relative peak intensities (first peak is always '1', twin-peak for pi-symmetry is x'rel' )
    by LW 12/16/2022 according to https://onlinelibrary.wiley.com/doi/full/10.1002/app.50939
    """
    from scipy.special import gamma as scp_gamma
    I=[]
    for hh,h in enumerate(hw):
        a=h/(np.log(2)**(1/beta[hh]))
        mu=[xc[hh],xc[hh]-180,xc[hh]+180]
        for mm,m in enumerate(mu):
            s=1
            if mm != 0:
                s=rel[hh]
            I.append(s*beta[hh]/(2*a*scp_gamma(1/beta[hh]))*np.exp(-np.abs(x-m)**beta[hh]/(a**beta[hh])))
    return np.sum(I,axis=0)

def I_GND_(x,xc,hw,beta):
    """
    generalized normal distribution: can be used as order parameter distribution to extract order parameters from SAXS intensities
    I_GND(x,xc,hw,beta)
    x: angle [deg]
    xc: list of center position(s) of peak(s) [deg], NOTE: will automatically generate twin-peak for pi-symmetry
    hw: list of half width(s) of peak(s) [deg]
    beta: sharpness parameter: 2: wide, 1: sharp
    by LW 12/16/2022 according to https://onlinelibrary.wiley.com/doi/full/10.1002/app.50939
    """
    from scipy.special import gamma as scp_gamma
    I=[]
    for hh,h in enumerate(hw):
        a=h/(np.log(2)**(1/beta[hh]))
        mu=[xc[hh],xc[hh]-180,xc[hh]+180]
        for m in mu:
            I.append(beta[hh]/(2*a*scp_gamma(1/beta[hh]))*np.exp(-np.abs(x-m)**beta[hh]/(a**beta[hh])))
    return np.sum(I,axis=0)

def I_LD(x,xc,hw,rel):
    """
    Lorentz distribution: can be used as order parameter distribution to extract order parameters from SAXS intensities
    I_LD(x,xc,hw)
    x: angle [deg]
    xc: list of center position(s) of peak(s) [deg], NOTE: will automatically generate twin-peak for pi-symmetry
    hw: list of half width(s) of peak(s) [deg]
    rel: list of relative peak intensities (first peak is always '1', twin-peak for pi-symmetry is x'rel' )
    by LW 12/16/2022 according to https://onlinelibrary.wiley.com/doi/full/10.1002/app.50939
    """
    I=[]
    for hh,h in enumerate(hw):
        mu=[xc[hh],xc[hh]-180,xc[hh]+180]
        for mm,m in enumerate(mu):
            s=1
            if mm != 0:
                s=rel[hh]
            I.append(s/(np.pi*h)*h**2/((x-m)**2+h**2))
    return np.sum(I,axis=0)

def I_GD(x,xc,hw):
    """
    Gaussian distribution: can be used as order parameter distribution to extract order parameters from SAXS intensities
    I_GD(x,xc,hw)
    x: angle [deg]
    xc: list of center position(s) of peak(s) [deg], NOTE: will automatically generate twin-peak for pi-symmetry
    hw: list of half width(s) of peak(s) [deg]
    rel: list of relative peak intensities (first peak is always '1', twin-peak for pi-symmetry is x'rel' )
    by LW 12/16/2022 according to https://onlinelibrary.wiley.com/doi/full/10.1002/app.50939
    """
    I=[]
    for hh,h in enumerate(hw):
        s=h/(2*np.log(2))**(1/2)
        mu=[xc[hh],xc[hh]-180,xc[hh]+180]
        for mm,m in enumerate(mu):
            s=1
            if mm != 0:
                s=rel[hh]
            I.append(s/(s*np.sqrt(2*np.pi))*np.exp(-np.abs(x-m)**2/(2*s**2)))
    return np.sum(I,axis=0)

def periodic_background(x,xoff,slope,constant):
    """
    function creates linear, but periodic (180deg) background
    periodic_background(x,xoff,slope,constant)
    xoff: shift along x (0<=xoff<=90)
    slope: slope of the linear background (switches sign to be periodic)
    constant: constant offset for all x
    by LW 12/16/2022
    """
    back=[];b=slope
    if xoff <0 or xoff>90:
        raise Exception('offset must be in the range [0,90] deg.!')

    for i in x:
        if i>xoff and i<=90+xoff: 
            back.append(b*i-b*xoff)
        elif i<=xoff:
            back.append(-b*i+b*xoff)
        elif i>180+xoff and i<=270+xoff:
            back.append(b*i-(180+xoff)*b)
        elif i>90+xoff and i<=180+xoff:
            back.append(-b*i+b*(180+xoff))
        elif i>270+xoff and i<=360+xoff:
            back.append(-b*i+b*(360+xoff))
    return np.array(back)+constant

def I_GND_fit_func(x,xc,hw,beta,rel,scale,xoff,slope,const):
    """
    helper function for fitting: combining Generalized Normal Distribution (GND) with periodic linear background
    Note: while I_GND can handle multiple peaks in [0,pi], this fit function can only take one peak position
    typically used as Intensity Distribution Function (IDF) in order parameter analysis
    by LW 12/16/2022
    """
    xc=[xc];hw=[hw];beta=[beta];rel=[rel]
    return scale*I_GND(x,xc,hw,beta,rel)+periodic_background(x,xoff,slope,const)

def I_GD_fit_func(x,xc,hw,rel,scale,xoff,slope,const):
    """
    helper function for fitting: combining Gaussian Distribution (GD) with periodic linear background
    Note: while I_GD can handle multiple peaks in [0,pi], this fit function can only take one peak position
    typically used as Intensity Distribution Function (IDF) in order parameter analysis
    by LW 12/17/2022
    """
    xc=[xc];hw=[hw];rel=[rel]
    return scale*I_GD(x,xc,hw,rel)+periodic_background(x,xoff,slope,const)

def I_LD_fit_func(x,xc,hw,rel,scale,xoff,slope,const):
    """
    helper function for fitting: combining Lorentzian Distribution (LD) with periodic linear background
    Note: while I_LD can handle multiple peaks in [0,pi], this fit function can only take one peak position
    typically used as Intensity Distribution Function (IDF) in order parameter analysis
    by LW 12/17/2022
    """
    xc=[xc];hw=[hw];rel=[rel]
    return scale*I_GD(x,xc,hw,rel)+periodic_background(x,xoff,slope,const)

def linear_intercept(x,a,b):
    return a*(x-b)
#######################################################################################################
# Non-mathematical functions:
#######################################################################################################

def get_uid_list(scan_uid_list,user=None,cycle=None,uid_length=99,fail_nonexisting=True,verbose=False,warning=True):
    """
    convert list of scan_ids to uids (with additional optional information 'user' and 'cycle') or vice versa
    get_uid_list(scan_uid_list,user=None,cycle=None)
    scan_uid_list: can be a mixed list of scan_ids [integers] and uids or short uids [strings]
    user: user name associated with the data [string]
    fail_nonexisting: if True -> raises Exception if a combination of scan_id/uid and cycle/user cannot be found; if False -> skips scan_id/uid, in this case the returned lists are shorter than the input list
    returns: scan_list, uid_list (uid[:uid_length] -> can return shortened uid strings)
    verbose: if True -> print pairs of scan_id, uid
    warning: if True -> print warning when combination of user, cycle and scan_id cannot be found (could be solved more elegantly with different verbose levels...)
    LW 01/03/2024
    """
    u_list = []; s_list=[];message_str = ''
    #for s in scan_uid_list:
    for s in tqdm (scan_uid_list,desc="getting scan_ids/uids…",  ascii=False, ncols=200,file=sys.stdout, colour='GREEN'):
        try:
            if type(s) == int:
                #s_list.append(s)
                if not user and not cycle:
                     u=[h.start['uid'] for h in db(scan_id=s)][0]
                elif user and not cycle:
                    u=[h.start['uid'] for h in db(scan_id=s, user=user)][0]
                elif not user and cycle:
                    u=[h.start['uid'] for h in db(scan_id=s, cycle=cycle)][0]
                elif user and cycle:
                    u=[h.start['uid'] for h in db(scan_id=s, user=user,cycle=cycle)][0]
                u_list.append(u[:uid_length])
                s_list.append(s)
            else:       
                h=db[s];u_list.append(h.start['uid'][:uid_length]);s_list.append(h.start['scan_id'])
        except:
            if fail_nonexisting: raise Exception('ERROR: combination of scan_id/uid: %s and user=%s and cycle=%s could not be found...'%(s,user,cycle))
            else: 
                if warning:
                    message_str+='WARNING: combination of scan_id/uid: %s and user=%s and cycle=%s could not be found...-> skip (output list of uids will be shorter than input list!)\n'%(s,user,cycle)
                    #print('WARNING: combination of scan_id/uid: %s and user=%s and cycle=%s could not be found...-> skip (output list of uids will be shorter than input list!)'%(s,user,cycle))
    print(message_str)
    if verbose:
        print('detailed information for all scan_ids:')
        for ss,s in enumerate(s_list):
            print('uid list: %s  ->  scan list: %s'%(u_list[ss],s))
    return s_list,u_list

import pickle
def save_pickle(filepath,data_dict,verbose=False):
    """
    wrapper function to save python dictionaries as pkl files
    save_pickle(filepath,data_dict,verbose=False)
    filepath: [string] path to save the data, including filename (IF ending is NOT .pkl, it will be added)
    data_dict: dictionary with data to save
    LW 01/03/2024
    """
    if filepath[-4:] != '.pkl':
        filepath+='.pkl'
    with open(filepath, 'wb') as dict_items_save:
            pickle.dump(data_dict, dict_items_save)
    if verbose:
        print('Data has been saved to file: '+filepath)
        
def load_pickle(filepath,verbose=False):
    """
    wrapper function to load content of .pkl files as python dictionaries
    load_pickle(filepath,verbose=False)
    filepath: [string] path to load the data, including filename (IF ending is NOT .pkl, it will be added)
    returns dictionary with data in dictionary form from pkl file
    LW 01/03/2024
    """
    if filepath[-4:] != '.pkl':
        filepath+='.pkl'
    if verbose:
        print('Loading data from file: '+filepath)
    with open(filepath, 'rb') as dict_items_open:
        return pickle.load(dict_items_open)

def get_data_dir(uid,_base_path_,_base_path_pass_,create=False,verbose=False):
    """
    returns for a given uid
    '/nsls2/data/chx/legacy/analysis/cycle/user/uid/'
    or
    '/nsls2/data/chx/proposals/cycle/pass-123456/uid/'
    """
    md=db[uid].start
    if 'proposal' in md.keys():
        data_dir = _base_path_pass_+'%s/pass-%s/Results/%s/'%(md['cycle'],md['proposal']['proposal_id'],md['uid'])
    else:
         data_dir = _base_path_+'%s/%s/Results/%s/'%(md['cycle'],md['user'],md['uid'])
    if verbose:
        print('data directory for uid: %s: %s'%(md['uid'],data_dir))
    if create:
        os.mkdir(data_dir)
        if verbose:
            print('created directory %s!'%data_dir)
    return data_dir

def get_result_dir(uid,_base_path_,_base_path_pass_,create=False,verbose=False):
    """
    returns for a given uid
    '/nsls2/data/chx/legacy/analysis/cycle/user/Results/'
    or
    '/nsls2/data/chx/proposals/cycle/pass-123456/Results/'
    """
    md=db[uid].start
    if 'proposal' in md.keys():
        result_dir = _base_path_pass_+'%s/pass-%s/Results/'%(md['cycle'],md['proposal']['proposal_id'])
    else:
         result_dir = _base_path_+'%s/%s/Results/'%(md['cycle'],md['user'])
    if verbose:
        print('result directory for uid: %s: %s'%(md['uid'],data_dir))
    if create:
        os.mkdir(result_dir)
        if verbose:
            print('created directory %s!'%result_dir)
    return result_dir

def get_pass_dir(uid,_base_path_,_base_path_pass_,create=False,verbose=False):
    """
    returns for a given uid
    '/nsls2/data/chx/legacy/analysis/cycle/user/'
    or
    '/nsls2/data/chx/proposals/cycle/pass-123456/'
    """
    md=db[uid].start
    if 'proposal' in md.keys():
        result_dir = _base_path_pass_+'%s/pass-%s/'%(md['cycle'],md['proposal']['proposal_id'])
    else:
         result_dir = _base_path_+'%s/%s/'%(md['cycle'],md['user'])
    if verbose:
        print('result directory for uid: %s: %s'%(md['uid'],data_dir))
    if create:
        os.mkdir(result_dir)
        if verbose:
            print('created directory %s!'%result_dir)
    return result_dir

def monitor_resources(total_time=1E4,update_time=5):
    """
    monitor computing resources (cpu and memory usage, free memory) and displays a plot
    total_time: total time [s] for this monitor to run
    update time: specify how often to collect resource data [s]
    plot will be updated after each 10th update   
    """
    import psutil
    from datetime import datetime
    for i in range(5000):
        time_stamp.append(time.time())
        cpu_usage.append(psutil.cpu_percent());memory_usage.append(psutil.virtual_memory().percent);memory_free.append(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
        time.sleep(5)
    
        if np.mod(i,10) == 0:
            display(clear=True,wait=True)
            print('last updated: %s'%datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            fig,ax=plt.subplots(figsize=(5,4))
            now=time.time()
            plt.plot((now-np.array(time_stamp))/60,cpu_usage,label='cpu [%]')
            plt.plot((now-np.array(time_stamp))/60,memory_usage,label='memory [%]')
            plt.xlabel('minutes ago');plt.ylabel('cpu / memory usage [%]')
            plt.grid(True,which='both');plt.legend(loc='upper left',bbox_to_anchor=(1.1,1))
            ax=plt.gca()
            ax2=plt.twinx(ax)
            ax2.plot((now-np.array(time_stamp))/60,memory_free,'r-',label='memory [%]')
            ax2.set_ylabel('free memory [%]',color='r');ax2.set_ylim(0,100)
            display(plt.gcf())
            plt.close()