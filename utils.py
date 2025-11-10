from io import TextIOWrapper
import numpy as np
import sys

def read_fvecs(filename,limit:int=10000):
    """
    Reads a .fvecs file.
    Format: [int32 d] [float32 x d]
    Total bytes: 4 + d * 4
    """
    # Read all as int32
    a = np.fromfile(filename, dtype='int32')
    if a.size == 0:
        return np.array([], dtype=np.float32).reshape(0, 0)
        
    d = a[0]
    # d_dim + d_data = (4 + d*4) bytes / 4 bytes per int32 = 1 + d
    return a.reshape(-1, d + 1)[:limit, 1:].copy().view('float32')

def read_ivecs(filename):
    """
    Reads a .ivecs file.
    Format: [int32 d] [int32 x d]
    Total bytes: 4 + d * 4
    """
    # Read all as int32
    a = np.fromfile(filename, dtype='int32')
    if a.size == 0:
        return np.array([], dtype=np.int32).reshape(0, 0)
        
    d = a[0]
    # d_dim + d_data = (4 + d*4) bytes / 4 bytes per int32 = 1 + d
    return a.reshape(-1, d + 1)[:, 1:].copy().view('int32')

def read_bvecs(filename):
    """
    Reads a .bvecs file.
    Format: [int32 d] [uint8 x d]
    Total bytes: 4 + d * 1
    
    We must read this vector-by-vector because the
    component size (1 byte) doesn't match the
    dimension size (4 bytes).
    """
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            # Read the dimension (4 bytes)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break # End of file
            
            d = np.frombuffer(dim_bytes, dtype='int32')[0]
            
            # Read the data (d bytes)
            data_bytes = f.read(d)
            if len(data_bytes) != d:
                raise IOError("Incomplete vector data at end of file.")
                
            data = np.frombuffer(data_bytes, dtype='uint8')
            vectors.append(data)
            
    if not vectors:
        return np.array([], dtype=np.uint8).reshape(0, 0)
        
    return np.array(vectors)

dist_functions={}

dist_functions["l2"]="<->"
dist_functions["ip"]="<#>"
dist_functions["cosine"]="<=>"
dist_functions["l1"]="<+>"
# dist_functions["hamming"]="<~>"
# dist_functions["jaccard"]="<%>"


class clogger:
    def __init__(self,path:str):
        self._path:str=path
        self._file:TextIOWrapper=open(path,"w")
        self._context:str="unspecified"
    def close(self):
        self._file.close()
    def set_context(self,context:str):
        self._context=context

    def write(self,obj):
        self._file.write(f"[{self._context}] {str(obj)}\n")


def recall(truth,pred):
    recall = {}
    for key in truth.keys():
        corrects = 0
        recall = 0
        for true_vecs,pred_vecs in zip(truth[key],pred[key]):
            corrects = len(set(true_vecs) & set(pred_vecs))
            recall += corrects/len(true_vecs)
        recall = recall/len(truth[key])
