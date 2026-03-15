def get_roi_nr(qdict,q,phi,q_nr=True,phi_nr=False,q_thresh=0, p_thresh=0, silent=True, qprecision=5):
    """
    function to return roi number from qval_dict, corresponding  Q and phi, lists (sets) of all available Qs and phis
    [roi_nr,Q,phi,Q_list,phi_list]=get_roi_nr(..)
    calling sequence: get_roi_nr(qdict,q,phi,q_nr=True,phi_nr=False, verbose=True)
    qdict: qval_dict from analysis pipeline/hdf5 result file
    q: q of interest, can be either value (q_nr=False) or q-number (q_nr=True)
    q_thresh: threshold for comparing Q-values, set to 0 for exact comparison
    phi: phi of interest, can be either value (phi_nr=False) or q-number (phi_nr=True)
    p_thresh: threshold for comparing phi values, set to 0 for exact comparison
    silent=True/False: Don't/Do print lists of available qs and phis, q and phi of interest
    by LW 10/21/2017
    update by LW 08/22/2018: introduced thresholds for comparison of Q and phi values (before: exact match required)
    update 2019/09/28 add qprecision to get unique Q
    update 2020/3/12 explicitly order input dictionary to fix problem with environment after 2019-3.0.1
    """
    import collections
    from collections import OrderedDict
    qdict = collections.OrderedDict(sorted(qdict.items()))
    qs=[]
    phis=[]
    for i in qdict.keys():
        qs.append(qdict[i][0])
        phis.append(qdict[i][1])  
    qslist=list(OrderedDict.fromkeys(qs))
    qslist = np.unique( np.round(qslist, qprecision ) )
    phislist=list(OrderedDict.fromkeys(phis))
    qslist=list(np.sort(qslist))
    phislist=list(np.sort(phislist))
    if q_nr:
        qinterest=qslist[q]
        qindices = [i for i,x in enumerate(qs) if np.abs(x-qinterest) < q_thresh]
    else:
        qinterest=q
        qindices = [i for i,x in enumerate(qs) if np.abs(x-qinterest) < q_thresh] # new
    if phi_nr:
        phiinterest=phislist[phi]
        phiindices = [i for i,x in enumerate(phis) if x == phiinterest]
    else:
        phiinterest=phi
        phiindices = [i for i,x in enumerate(phis) if np.abs(x-phiinterest) < p_thresh] # new
    ret_list=[list(set(qindices).intersection(phiindices))[0],qinterest,phiinterest,qslist,phislist] #-> this is the original
    if silent == False:
        print('list of available Qs:')
        print(qslist)
        print('list of available phis:')
        print(phislist)
        print('Roi number for Q= '+str(ret_list[1])+' and phi= '+str(ret_list[2])+': '+str(ret_list[0]))
    return ret_list