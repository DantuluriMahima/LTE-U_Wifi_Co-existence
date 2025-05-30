import math

class PARAMS:

    scene = 1
    numofLTEBS = 1
    numofWifiBS = 1
    numofLTEUE = 90
    numofWifiUE = 15
    
    times_frames = 10 # Simulate for this value x 10ms
    test_eps = 50

    vary_from = 400000000   # Change of load at this iteration number, give smaller value than times_frames if you want to vary the load
    vary_load = 0   # this flag is used in runtime, do not set this
    vary_for_every = 0    # Load will be changed for these many frame iterations

    # Load changed to given set of users, Fully functional load change is not yet implemented
    # Limited to only one change.
    set_users_LTE  = [20]
    set_users_Wifi = [5]

    vary_iterator = 0   # iterates in the set user list

    # Deplpyment area
    length = 100
    breadth = 100
    
    prob = 0.5  # Probability of transmission

    pTxWifi = .19    # Unit: Watt
    pTxLTE = [0.19,0.19,0.19]  # Unit: Watt
    noise = -80 # Noise in environment
    Gain_antenna = 14 #Unit : db Usually in range 12-16db
    Gain_receiver = 1 #UNit : bd Usually in range 0-2db
    duration_frame = 10 # 10 ms
    duration_subframe = 1   # 1 ms
    duration_slot = 0.5 # 0.5 ms

    duration_wifislot = 9   # 9 us

    wifi_slots_per_subframe = math.floor((duration_subframe*1000)/duration_wifislot) #Change to *1000

    PRB_symbols = 7 # this is 7 symbols per resource block
    PRB_subcarriers = 12 # this is 12 subcarriers per resource block (derived value)
    
    PRB_total_symbols = PRB_subcarriers*PRB_symbols # Symbols in one PRB

    PRB_subcarrier_bandwidth = 15 # KHz
    PRB_bandwidth = 180 # KHz

    PRB_total_prbs = 100

    pTx_one_PRB = [0,0,0]

    # Profile of Users
    LTEprofiles = [256,512,1024,2048]
    Wifiprofiles = [256,512,1024,2048]
    LTE_ratios = [4,3,2,1]
    wifi_ratios = [1,2,3,4]
    # Lists to allocate profiles, do not edit
    LTE_profile_prob = []
    wifi_profile_prob = []
    LTE_profile_c_prob = []
    wifi_profile_c_prob = []

    # Seed value, now passed from command line while execution
    seed_valueLTE = 10
    seed_valueWifi = 10

    # MCS Table stored in dictionary
    LTE_MCS = {-6.936:0.1523,-5.147:0.2344,-3.18:0.377,-1.253:0.6016,0.761:0.877,2.699:1.1758,4.694:1.4766,6.525:1.9141,8.573:2.4063,10.366:2.7305,12.289:3.3223,14.173:3.9023,15.888:4.5234,17.814:5.1152,19.829:5.5547}
    wifi_MCS = {2:7.2, 5:14.4, 9:21.7, 11:28.9, 15:43.3, 18:57.8, 20:65.0, 25:72.2, 29:86.7}

    # Backoff range for CSMA/CA
    # in slots
    backoff_lower = 5
    backoff_upper = 20

    SIFS = 1    # Slots
    DIFS_slots = 2 + SIFS    # Slots

    # Conversion functions, may have incorrect name or remain unused
    
    def get_bits_per_slot_from_Kbps(self,value_Kbps):

        return (value_Kbps/2)
    
    def get_bits_per_wifi_slot_from_Mbps(self,value_Mbps):

        # Ex: 128 Mbps to bits per wifi slot (bits/9 us)
        # 128 * 10^6 bits 10^-6 us
        # 128  * 9 (bits/nine-us)

        return (value_Mbps*9)

    def get_dB_from_dBm(self,value_dBm):
        return (value_dBm-30)

    def get_dBm_from_Watt(self,value_watt):
        return (10*math.log(value_watt*1000,10))

    def get_dB_from_Watt(self,value_watt):
        val_dB = 10*math.log(value_watt,10)
        return val_dB
        
    def get_Watt_from_dB(self,value_dB):
        val_Watt = 10**(value_dB/10)
        return val_Watt

    def get_mWatt_from_dBm(self,value_dBm):
        val_mWatt = 10**(value_dBm/10)
        return val_mWatt/1000