import numpy as np
from Params.ConstantParams import PARAMS

class LTEBaseStation:
    bsID: int
    x: int  # x-coordinate
    y: int  # y-coordinate
    pTx: float    # Transmission Power in watt
    
    pTx_antenna = PARAMS.pTxLTE   # Transmission Power for 3 antennas (in watt)
    SINR_antenna = np.array([None, None, None])   # SINR for 3 antennas
    user_list = np.array([])
    user_list_antenna = [np.array([]), np.array([]), np.array([])]  # Separate user lists for each antenna
    t_user_list_antenna = [np.array([]), np.array([]), np.array([])]
    # To keep track of active users per antenna
    lusscount_antenna = [None, None, None]
    directions = np.array([0, 120, 240])  # Directions in degrees
    
    bits_per_symbol_of_user = dict()
    
class WifiBaseStation:
    bsID: int
    x: int  # x-coordinate
    y: int  # y-coordinate
    pTx: float    # Transmission Power in watt
    pTx = PARAMS.pTxWifi
    user_list = np.array([])  # List of users associated with this BaseStation. Exploiting Python's feature to assign objects to variables, thus avoiding Circular Dependency between BS and UE
    t_user_list = np.array([])
    wusscount = None
    SNR=None
    #format = None
    
    bits_per_symbol_of_user = dict()