def get_sid_filenames(hdr,verbose=False):
    import glob
    from time import strftime, localtime
    start_doc = hdr.start
    stop_doc = hdr.stop
    success = False
    
    ret = (start_doc["scan_id"], start_doc["uid"], glob.glob(f"{start_doc['data path']}*_{start_doc['sequence id']}_master.h5")) # looking for (eiger) datafile at the path specified in metadata
    if len(ret[2])==0:
        if verbose: print('could not find detector filename from "data_path" in metadata: %s'%start_doc['data path'])
    else:
         if verbose: print('Found detector filename from "data_path" in metadata!');success=True
    
    if not success: # looking at path in metadata, but taking the date from the run start document
        data_path=start_doc['data path'][:-11]+strftime("%Y/%m/%d/",localtime(start_doc['time']))
        ret = (start_doc["scan_id"], start_doc["uid"], glob.glob(f"{data_path}*_{start_doc['sequence id']}_master.h5"))
        if len(ret[2])==0:
            if verbose: print('could not find detector filename in %s'%data_path)
        else:
             if verbose: print('Found detector filename in %s'%data_path);success=True

    if not success: # looking at path in metadata, but taking the date from the run stop document (in case the date rolled over between creating the start doc and staging the detector)
        data_path=start_doc['data path'][:-11]+strftime("%Y/%m/%d/",localtime(stop_doc['time']))
        ret = (start_doc["scan_id"], start_doc["uid"], glob.glob(f"{data_path}*_{start_doc['sequence id']}_master.h5"))
        if len(ret[2])==0:
            if verbose: print('Sorry, could not find detector filename....')
        else:
             if verbose: print('Found detector filename in %s'%data_path);success=True
    return ret 