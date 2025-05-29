from collections import defaultdict
import numpy as np
import h5py

def extract_measurement_truth_lists(file_name, tpc_filter=1):
    def extract_sparse_list(group):
        grid = group['grid_index'][:]
        charge = group['charge'][:]
        tpc_id = group['tpc_id'][:]
        event_id = group['event_id'][:]
        data = defaultdict(lambda: defaultdict(list))
        for i in range(len(charge)):
            tid = tpc_id[i]; eid = event_id[i]
            #if tid != tpc_filter: continue
            y = int(grid[i, 0])
            z = int(grid[i, 1])
            t = int(grid[i, 2])
            adc = float(charge[i])
            data[eid][(y, z)].append((t, adc))
        return data  # event_id → {(y,z): [(t,adc)]}

    with h5py.File(file_name, 'r') as f:
        hits_raw = extract_sparse_list(f['hits'])   # measured
        truth_raw = extract_sparse_list(f['effq'])  # true Q

    measurements = []  # List[ List[((y,z), t, adc)] ]
    truths = []        # List[ Dict[(y,z)] = Q ]
    
    for eid in sorted(hits_raw):
        pix_adc = hits_raw[eid]
        flat = [ ((y, z), t, adc) for (y, z), seq in pix_adc.items() for (t, adc) in seq ]
        measurements.append(flat)

        qtruth = {}
        for (y, z), seq in truth_raw.get(eid, {}).items():
            qtruth[(y, z)] = sum(adc for _, adc in seq)
        truths.append(qtruth)

    return measurements, truths
        
