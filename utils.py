import numpy as np

def read_fvecs(filename):
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
    return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')

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