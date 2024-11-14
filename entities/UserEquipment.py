import numpy as np
import math
import random
from Params.ConstantParams import PARAMS
from entities.BaseStation import WifiBaseStation
from entities.BaseStation import LTEBaseStation

class UserEquipment:
    #LTE part
    ueID: int
    x: int  # x-coordinate
    y: int  # y-coordinate
    powerRcvd_antenna_list = np.array([None, None, None])  # List of received powers from 3 directional antennas
    LTE_bs = None  # the BaseStation to which this UE is connected
    best_antenna_index = None  # The index of the best directional antenna (0, 1, 2)
    SINR = None
    req_LTE_data_rate = None  # required data rate in Kbps
    req_LTE_bits_per_slot = None
    req_LTE_no_PRB = None  # PRBs required by user
    transmission_finished = 0
    bits_sent = 0
    
    def getLTEPowerRcvd(self, user, b, antenna_index):
        """Calculate the received power from a specific directional antenna with 120-degree beamwidth."""
        dist = ((b.x - user.x) ** 2 + (b.y - user.y) ** 2) ** 0.5
        Gain_antenna = PARAMS.Gain_antenna
        Gain_receiver = PARAMS.Gain_receiver
        
        pathloss = 20 * math.log(2400, 10) + 30 * math.log(dist, 10) + 0 - 28 - Gain_antenna - Gain_receiver
        
        # Adjust power received based on antenna gain
        prcvd = PARAMS().get_dB_from_Watt(b.pTx_antenna[antenna_index]) - pathloss
        self.powerRcvd_antenna_list[antenna_index] = np.append(self.powerRcvd_antenna_list[antenna_index],prcvd)

        return prcvd


    def measureSINRandConnect(self, lbss, wbss):
        """Measure SINR for all LTE base stations and connect to the best base station and antenna based on the angle."""
        maxsinr = -99999999
        max_bs = None
        max_antenna_index = None
        wifi_power_sum = 0

        # Summing up received power from WiFi base stations (for interference calculation)
        for w in wbss:
            w_user = WifiUserEquipment()
            wifi_power_recv = w_user.getWifiPowerRcvd(w, self)
            wifi_power_sum += PARAMS().get_Watt_from_dB(wifi_power_recv)

        wifi_part = wifi_power_sum + PARAMS().get_mWatt_from_dBm(PARAMS().noise)

        # Iterate over LTE base stations to find the best base station based on SINR
        for b in lbss:
            antenna_powers = []  # To store the received powers from all antennas
            for antenna_index in range(3):
                lte_power_rcvd = self.getLTEPowerRcvd(self, b, antenna_index)
                antenna_powers.append(PARAMS().get_Watt_from_dB(lte_power_rcvd))

            # Determine the user's location and the relevant antenna sector
            dx = self.x - b.x
            dy = self.y - b.y
            angle = math.degrees(math.atan2(dy, dx)) % 360  # Angle in degrees (0 to 360)

            # Find the appropriate antenna based on the angle
            if 0 <= angle < 120:
                main_antenna_index = 0  # Antenna 0
            elif 120 <= angle < 240:
                main_antenna_index = 1  # Antenna 1
            else:
                main_antenna_index = 2  # Antenna 2

            interference_power = 0
            Attenuation_factor = 10 ** (-30 / 10)  # -30 dB attenuation for antennas outside the sector

            for antenna_index in range(3):
                if antenna_index != main_antenna_index:
                    # Apply attenuation to antennas outside the user's sector
                    interference_power += antenna_powers[antenna_index] * Attenuation_factor
            # Calculate SINR for the main antenna
            sinr_temp = PARAMS().get_dB_from_Watt(antenna_powers[main_antenna_index]) - PARAMS().get_dB_from_Watt(interference_power + wifi_part)

            #print(f"Base station {b.bsID}, Antenna {main_antenna_index}: SINR = {sinr_temp}, User: {self.ueID}")  # Debug SINR values

            if sinr_temp > maxsinr:
                maxsinr = sinr_temp
                max_bs = b  # Best base station
                max_antenna_index = main_antenna_index  # Best antenna

        # If a best base station is found, assign the user to it
        if max_bs is not None:
            self.LTE_bs = max_bs  # Connect the user to the best base station
            #print(f"User {self.ueID} assigned to base station {max_bs.bsID}")  # Debug base station assignment

            self.SINR = maxsinr
            self.best_antenna_index = max_antenna_index
            max_bs.SINR_antenna[max_antenna_index] = maxsinr

            # Add user to the appropriate antenna's user list at the base station
            self.LTE_bs.user_list = np.append(self.LTE_bs.user_list, self)  # Add user to base station's main user list
            self.LTE_bs.user_list_antenna[self.best_antenna_index] = np.append(self.LTE_bs.user_list_antenna[self.best_antenna_index], self)

            #print(f"User {self.ueID} assigned to antenna {self.best_antenna_index} of base station {self.LTE_bs.bsID}")  # Debug antenna assignment

        if maxsinr <= -6.936:
            self.SINR = -6.936
            
        return max_bs, max_antenna_index

    
        
class WifiUserEquipment:
    ueID: int
    x: int  # x-coordinate
    y: int  # y-coordinate
    powerRcvd_list = np.array([])  # List of users associated with this BaseStation
    bs = None # the BS to which this UE is connected. Exploiting Python's feature to assign objects to variables, thus avoiding Circular Dependency between BS and UE
    SNR = None
    probability = None  # Probability with which the user transmits
    # wifislotsreq = None
    best_lte_region = None
    req_data_rate = None # required data rate in Kbps
    req_bits_per_slot = None
    
    req_no_wifi_slot = None   # total wifi slots required by user = (required bits per wifi slot)
    #                                                               / (bits per wifi slot)
    random_backoff_slots = 0
    random_backoff_flag = 0
    # busy_count = 0
    channel = None
    #SHOULD CHANGE TO 0
    DIFS_flag = 0
    DIFS_slots = PARAMS().DIFS_slots

    RTS_flag=0

    bits_sent = 0

    def getWifiPowerRcvd(self,b,user):
        dist = float()
        dist = ((b.x-user.x)**2 + (b.y-user.y)**2 )**0.5

        pathloss = float()

        pathloss=20*math.log(2400,10)+30*math.log(dist,10)+0-28

        #Measure power
        prcvd = float()
        prcvd = PARAMS().get_dB_from_Watt(b.pTx) - pathloss

        return prcvd
    

    # Measure SNR from all Base Stations, Assign max BS to self UE and return index of max BS
    def measureSNRandConnect(self,lbss,wbss):
        
        maxsnr = -99999999
        maxind = 0
        ind = 0
        snr_list = []

        for w in wbss:
            
            wifi_power_rcvd = self.getWifiPowerRcvd(w,self)

            snr_temp = wifi_power_rcvd-(PARAMS().get_dB_from_dBm(PARAMS().noise))

            snr_list.append(snr_temp)

            if maxsnr <= snr_temp:
                maxsnr = snr_temp
                maxind = ind

            ind+=1
        if lbss is not None:
            self.label_wifi_interference_by_angle(lbss)

        self.SNR = maxsnr
        self.bs = wbss[maxind]
        #print(f"User {self.ueID} has SNR {self.SNR}")
        self.bs.user_list = np.append(wbss[maxind].user_list, self)
        return maxind


#### IG THIS WORKS

    def label_wifi_interference_by_angle(self, lbss):
        """Label WiFi user based on proximity and angle to nearest LTE base station.
        Determine if user is in a segment for antenna 0, 1, 2, or in an overlap region.
        """
        
        closest_bs = None
        min_distance = float('inf')
        interference_threshold = 0  # Threshold for interference in dB
        self.best_lte_region = "No significant interference"  # Default label

        # Find the closest LTE base station
        for lte_bs in lbss:
            dist = math.sqrt((lte_bs.x - self.x) ** 2 + (lte_bs.y - self.y) ** 2)
            if dist < min_distance:
                min_distance = dist
                closest_bs = lte_bs

        # Proceed if we found a nearby base station
        if closest_bs:
            # Calculate the angle from the WiFi user to the closest base station
            angle_to_bs = math.degrees(math.atan2(closest_bs.y - self.y, closest_bs.x - self.x)) % 360

            # Check which antenna segment the angle falls into
            if 15 <= angle_to_bs < 105:
                self.best_lte_region = "0"  # Within antenna 0's main sector
            elif 135 <= angle_to_bs < 225:
                self.best_lte_region = "1"  # Within antenna 1's main sector
            elif 255 <= angle_to_bs < 345:
                self.best_lte_region = "2"  # Within antenna 2's main sector
            
            # Determine if the user is in an overlapping region between antennas
            if 105 <= angle_to_bs < 135:
                self.best_lte_region = "01"  # Overlap region of antennas 0 and 1
            elif 225 <= angle_to_bs < 255:
                self.best_lte_region = "12"  # Overlap region of antennas 1 and 2
            elif 345 <= angle_to_bs < 360 or 0 <= angle_to_bs <= 15:
                self.best_lte_region = "20"  # Overlap region of antennas 2 and 0

            #print(f"User {self.ueID} interference label: Assigned to LTE antenna segment {self.best_lte_region}")

    def setRandomBackoff(self):
        self.random_backoff_slots = random.randint(PARAMS().backoff_lower, PARAMS().backoff_upper)
