import numpy as np
from Params.ConstantParams import PARAMS
from tqdm import tqdm
from entities.BaseStation import LTEBaseStation, WifiBaseStation
from entities.UserEquipment import UserEquipment
from run.ServiceClass import ServiceClass, GraphService
from Qlearning.Learning import Learning
import copy
import math
from itertools import chain

def bringRealUser(selected_user,wuss):

    for u in wuss:
        if u.ueID == selected_user.ueID:
            return u


def generate_format(current_state):
    """Generate a 3x10 format based on the first three elements of the current state (l, m, n)."""
    l, m, n = current_state
    format_array = []

    # Define each row based on the values of l, m, and n
    for zeros_count in (l, m, n):
        # Create a row with `zeros_count` zeros followed by `10 - zeros_count` ones
        row = [0] * zeros_count + [1] * (10 - zeros_count)
        format_array.append(row)
    
    return format_array

def main():
    # Initialize parameters
    F = []
    LTE_T = []
    LTE_P = []
    E = []
    U =[]
    Uw =[]
    LUS = []
    WUS = []
    W_T = []
    for i in range(100):
        scene_params = PARAMS()
        graphservice = GraphService()
        # Create service instance
        service = ServiceClass()

        # Scene number for creating base stations and users
        # Scene0: numofLTEs & numofWi-Fis as declared in PARAMS
        # Scene1: 1 LTE & 1 Wi-Fi (colocated)

        # ---Not updated----
        # Scene2: 1 LTE & 1 Wi-Fi (apart)
        # Scene3: 3 LTE & 3  Wi-Fi
        # Scene4: 1 LTE & 3 Wi-Fi
        # Scene5: 3 LTE & 1 Wi-Fi
        #--------------------------
        scenenum = 1
        description = "Not Random-Generic"   #vary_load and stuff not still Written DO NOT FORGET!!!!!!!!!!!!!!
        # Create LTE Base Stations
        lte_bs_list = service.createLTEBaseStations(scene_params, scenenum)
        print("LTE Base Stations:")
        for bs in lte_bs_list:
            print(f"ID: {bs.bsID}, Location: ({bs.x}, {bs.y}), pTx: {bs.pTx}")

        # Create WiFi Base Stations
        wifi_bs_list = service.createWifiBaseStations(scene_params, scenenum)
        print("\nWiFi Base Stations:")
        for bs in wifi_bs_list:
            print(f"ID: {bs.bsID}, Location: ({bs.x}, {bs.y}), pTx: {bs.pTx}")

        # Create LTE Users
        LTE_users_list = service.createLTEUsers(scene_params,lte_bs_list[0],wifi_bs_list[0])
        print("\nLTE Users:")
        for user in LTE_users_list:
            print(f"ID: {user.ueID}, Location: ({user.x}, {user.y})")

        # Create WiFi Users
        wifi_users_list = service.createWifiUsers(scene_params,lte_bs_list[0],wifi_bs_list[0])
        print("\nWiFi Users:")
        for user in wifi_users_list:
            print(f"ID: {user.ueID}, Location: ({user.x}, {user.y})")

        for user in LTE_users_list:
            user.measureSINRandConnect(lte_bs_list, wifi_bs_list)  # Assume this connects the user to a base station
            user_index = np.where(LTE_users_list == user)[0]
            if user.SINR == -1:
                print(user_index)
                LTE_users_list = np.delete(LTE_users_list,user_index)
        for user in wifi_users_list:
            user.measureSNRandConnect(lte_bs_list, wifi_bs_list)  # Assume this connects the user to a base station
            user_index = np.where(wifi_users_list == user)[0]
            if user.SNR == -1:
                print(user_index)
                wifi_users_list = np.delete(wifi_users_list,user_index)

        # Print connected users to their respective base stations for LTE
        print("\nLTE Users connected to Base Stations:")
        for b in lte_bs_list:
            print(f"Base Station {b.bsID} at ({b.x}, {b.y}) is connected to users:")
            connected_users = b.user_list
            for user in connected_users:
                print(f"  User ID: {user.ueID} (Antenna Index: {user.best_antenna_index})")

        # Print connected users to their respective base stations for WiFi
        print("\nWiFi Users connected to Base Stations:")
        for b in wifi_bs_list:
            print(f"WiFi Base Station {b.bsID} at ({b.x}, {b.y}) is connected to users:")
            for user in b.user_list:
                print(f"  User ID: {user.ueID} in Lte antenna {user.best_lte_region}")

        service.calculate_profile_prob(scene_params)

        print("\n=== Profile and Probability Info ===")
        print("DataRate LTE Profiles: ",scene_params.LTEprofiles)
        print("DataRate Wifi Profiles: ",scene_params.Wifiprofiles)
        print("LTE user ratio:    ",scene_params.LTE_ratios)
        print("Wifi user ratio:   ",scene_params.wifi_ratios)
        print("\nLTE Prob: ",scene_params.LTE_profile_prob)
        print("LTE Cumulative Prob: ",scene_params.LTE_profile_c_prob)
        print("\nWifi Prob: ",scene_params.wifi_profile_prob)
        print("Wifi Cumulative Prob: ",scene_params.wifi_profile_c_prob)
        print("======")

        # Based on ratios decided by the user, assign data rates to UE
        service.assign_data_rate_to_users(scene_params, LTE_users_list, wifi_users_list)
        print("\n=== LTE user required data rates ===")
        for u in LTE_users_list:        
            print("LTE userid {}: {} Kbps".format(u.ueID,u.req_data_rate))
        print("======")
            

        
        print("\n=== Wifi user required data rates ===")
        for u in wifi_users_list:        
            print("Wifi userid {}: {} Kbps".format(u.ueID,u.req_data_rate))
        print("======")

        SINR = []
        SNR = []
        for u in LTE_users_list:
            #u.measureSINR(wifi_bs_list)
            SINR.append(u.SINR)

        service.decide_LTE_bits_per_symbol(lte_bs_list,scene_params)

        # for u in luss:

        #     print("LTE userid {}: {:.4f}".format(u.ueID,u.SINR))


        print("\n=== LTE BS Dictionary of User req bits ===")
        for b in lte_bs_list:
                
            users = b.user_list
            print("LTE BSid {}".format(b.bsID))
            for u in users:

                print(" @> {}:{}:{}".format(u.ueID,b.bits_per_symbol_of_user[u],u.best_antenna_index) )

        print("======")


        service.calculate_LTE_user_PRB(scene_params, LTE_users_list)

        
        print("\n=== LTE user SINR and MCS value ===")
        for b in lte_bs_list:
            for u in b.user_list:
                print("LTE Base Station {}: Antenna index {}: userid {}: SINR: {:.4f} @> {} bits per symbol".format(u.LTE_bs.bsID,u.best_antenna_index,u.ueID,u.SINR,b.bits_per_symbol_of_user[u]))
                print("Required PRBs: {}".format(u.req_no_PRB))
        
        print("\nTotal Required PRBs: {}".format(service.getTotalRequiredPRB(scene_params, LTE_users_list)))
        print("======")
        
        service.decide_wifi_bits_per_symbol(wifi_bs_list, scene_params)
        
        
        print("\n=== Wifi BS Dictionary of User req bits ===")
        for b in wifi_bs_list:
                
            users = b.user_list
            print("Wifi BSid {}".format(b.bsID))
            for u in users:

                print(" @> {}:{}".format(u.ueID,b.bits_per_symbol_of_user[u]))

        print("======")

        service.calculate_wifi_user_slots(scene_params, wifi_users_list)

        print("\n=== Wifi user SNR and MCS value ===")
        for b in wifi_bs_list:
            for u in b.user_list:
                print("Wifi userid {}: {:.4f} @> {} Mbps".format(u.ueID,u.SNR,b.bits_per_symbol_of_user[u]))
                print("Required wifi slots: {}".format(u.req_no_wifi_slot))

        print("\nTotal Required wifi slots: {}".format(service.getTotalRequiredWifiSlot(scene_params, wifi_users_list)))
        print("======")

        # Store total request by users
        ltereq = service.getTotalRequiredPRB(scene_params,LTE_users_list)
        wifireq, total = service.getTotalRequiredWifiSlot(scene_params,wifi_users_list)
        total_lte_req = 0
        for i in range(3):
            total_lte_req += ltereq[i]
        #Plotting graphs
        
        '''service.createLocationCSV(wifi_bs_list, lte_bs_list, LTE_users_list, wifi_users_list)
        graphservice.PlotScene(scenenum, description)
        graphservice.PlotHistSINR(SINR,scene_params)
        graphservice.PlotHistSNR(SINR,scene_params)'''

        # ====================== Simulation starts here ===============================
        # 1  -->Uplink / s / wifi
        # 0  -->Downlink / LTE
        # '0'/'1' --> SLOT
        rl = Learning()
        rl.load("trained_model.weights.h5")
        print("=============================================================================")

        Fairness = []   # Stores fairness for each frame combination
        LTE_Throughput = [] # Stores throughput of LTE
        LTE_Power = []

        Wifi_Throughput = [] # Stores total throughput of Wifi
        ECR = []
        Utilization = []
        Wifi_Utilization = []
        Frame_choosen = []
        LTE_User_satisfy = []
        Wifi_User_satisfy = []
        #print("LTE users: {} Wifi users: {}".format(LTE_users_list,wifi_users_list))

        channel_busy = [0,0,0,0,0,0]    # flag to denote channel busy

        # CSMA/CA stuff
        CTS = 0 # this is set to number of slots for which a wifi user gets CTS
        allwuss = [[], [], [], [], [], []]  # List of all waiting users per channel
        for user in wifi_users_list:
            if user.best_lte_region == "0":
                user.channel = 0
                allwuss[0].append(copy.deepcopy(user))
            elif user.best_lte_region == "1":
                user.channel = 1
                allwuss[1].append(copy.deepcopy(user))
            elif user.best_lte_region == "2":
                user.channel = 2
                allwuss[2].append(copy.deepcopy(user))
            elif user.best_lte_region == "01":
                user.channel = 3
                allwuss[3].append(copy.deepcopy(user))
            elif user.best_lte_region == "12":
                user.channel = 4
                allwuss[4].append(copy.deepcopy(user))
            elif user.best_lte_region == "20":
                user.channel = 5
                allwuss[5].append(copy.copy(user))
        tuserlist = [[],[],[],[],[],[]]  # Transmitting user list per channel
        RTSuserlist = [[],[],[],[],[],[]]  # RTS list per channel

        FinishedWifilist = []
        vary_for_every = [1]

        # if thisparams.vary_load == 1:
        
        vary_for_every = scene_params.vary_for_every
        
        Wifi_vary_factor = [1,1,1] #   initially set to 1
        LTE_vary_factor = [1,1,1]

        vary_for_every = scene_params.vary_for_every
        
        Wifi_vary_factor = 1 #   initially set to 1
        LTE_vary_factor = 1

        #lte_bs_list[0].format = format[rl.initial_state]
        print(rl.initial_state)
        rl.current_state = rl.initial_state
        rl.epsilon = 0.01

        for tf in tqdm(range(0,scene_params.times_frames)):
            
            if tf>scene_params.vary_from:
                scene_params.vary_load = 1

            rl.act(rl.current_state)
            print("\nChoosen Action: ",rl.current_action)
            print("New State: ",rl.current_state)
            rl.original_power = [0.19,0.19,0.19]
            lte_bs_list[0].format = generate_format(rl.current_state)
            
            lte_bs_list[0].pTx_antenna = scene_params.pTxLTE
            scene_params.pTx_one_PRB = [pTx / (2 * scene_params.PRB_total_prbs * scene_params.duration_frame * 100) for pTx in scene_params.pTxLTE]

            for lb in lte_bs_list:
                lb.bits_per_symbol_of_user = dict()
                
            service.decide_LTE_bits_per_symbol(lte_bs_list,scene_params)
            service.calculate_LTE_user_PRB(scene_params, LTE_users_list)
            print("Total Required PRBs: {}".format(service.getTotalRequiredPRB(scene_params, LTE_users_list)))
            
            LTECountS=[0,0,0]
            LTECountU=[0,0,0]

            WifiCountS=0
            WifiCountU=0

            LTEPowerS = 0.0

            total_PRBs = [0,0,0]  # Holds total PRBs allocated for LTE in 10ms
            total_Wifi_slots = 0    # Holds total slots allocated for Wifi in 10ms

            total_LTE_bits_sent = 0 # Holds bits sent by LTE users
            total_Wifi_bits_sent = 0 # Holds bits sent by Wifi users
            wifi_req = 0
            print("------------------------------------------------------------------------------------------- {}".format(tf))
            for subframe_iterator in range(10):
                single_zero = [0,0,0]
                multiple_zero = [0,0,0]
                all_one = [0,0,0]

                zero_counter = [0, 0, 0]  # Zero counters for each directional antenna (3 in total)
                one_counter = [0, 0, 0]   # One counters for each directional antenna (3 in total)

                lbs_single_transmission_ind = [None, None, None]

                # Initialize separate busy flags for each channel
                # Track busy status for each channel individually

                # Iterate over each LTE base station and each of their 3 directional antennas
                for b in lte_bs_list:
                    for antenna_index in range(3):  # Loop through each antenna channel
                        if b.format[antenna_index][subframe_iterator] == 0:
                            zero_counter[antenna_index] += 1
                        elif b.format[antenna_index][subframe_iterator] == 1:
                            one_counter[antenna_index] += 1
                idle_segments = []
                # Check each antenna channel for multiple zeros, single zero, or all ones
                for antenna_index in range(3):
                    if zero_counter[antenna_index] > 1:
                        multiple_zero[antenna_index] = 1
                        channel_busy[antenna_index] = 1  # Mark this channel as busy
                    elif zero_counter[antenna_index] == 1:
                        single_zero[antenna_index] = 1
                        channel_busy[antenna_index] = 1  # Mark this channel as busy
                        # Find the index of the base station with a single zero in this antenna channel
                        lbs_single_transmission_ind[antenna_index] = next(
                            i for i, b in enumerate(lte_bs_list) if b.format[antenna_index][subframe_iterator] == 0
                        )
                    elif one_counter[antenna_index] == len(lte_bs_list):
                        all_one[antenna_index] = 1
                        # Only add to total WiFi slots if the channel is idle (all ones)
                        idle_segments.append(str(antenna_index))
                        channel_busy[antenna_index] = 0  # Mark this channel as idle
                if one_counter[0] == 1 and one_counter[1] == 1 and one_counter[2] == 1:
                    total_Wifi_slots += 111
                
                for i in range(6):
                    Wifi_slots, total = service.getTotalRequiredWifiSlot(scene_params,allwuss[i])
                # Calculate WiFi slots based on idle segments and overlaps
                    wifi_req += total
                    total_Wifi_slots += total
                # "Simulation for one sub-frame (0/1) in a frame" ==============================
                ###################### HAVE TO CHANGE ALLWUSS AND RELATED STUFF AND ADD THE USER EQUIPMENT PARTS AND REAL USER FN FULL GA #################################
                Wifisensecount = 0
                # Function to get users from free channels
                def get_free_channel_users():
                    for i in range(3):
                        channel_busy[i] = 1 if (b.format[i][subframe_iterator] == 0) else 0
                    free_channels = [i for i in range(3) if channel_busy[i] == 0]
                    channel_busy[3] = 0 if (channel_busy[0] == 0 and channel_busy[1] == 0) else 1
                    channel_busy[4] = 0 if (channel_busy[2] == 0 and channel_busy[1] == 0) else 1
                    channel_busy[5] = 0 if (channel_busy[0] == 0 and channel_busy[2] == 0) else 1
                    # Add users assigned to the free channels
                    users_to_transmit = [[] for _ in range(6)]  # 6 empty sublists for each channel
        
                    # Add users assigned to the free channels
                    for ch in free_channels:
                        users_to_transmit[ch].extend(allwuss[ch])  # Add all users from the available channel
                    
                    # Add users from composite channels if they are free
                    if 1 in free_channels and 0 in free_channels:
                        users_to_transmit[3].extend(allwuss[3])  # Add users from composite channel 3
                    
                    if 1 in free_channels and 2 in free_channels:
                        users_to_transmit[4].extend(allwuss[4])  # Add users from composite channel 4
                    
                    if 2 in free_channels and 0 in free_channels:
                        users_to_transmit[5].extend(allwuss[5])  # Add users from composite channel 5
        
                    #print(channel_busy)

                    return users_to_transmit, free_channels
                rem_wifi_slots = scene_params.wifi_slots_per_subframe
                while Wifisensecount < scene_params.wifi_slots_per_subframe:

                    # Get the list of users who can transmit based on free channels
                    users_to_transmit, free_channels = get_free_channel_users()
                    flattened_users = [user for sublist in users_to_transmit for user in sublist]

                    #print(f"Free channels: {free_channels}, Users eligible for transmission: {[u.ueID for u in flattened_users]}")
                    if len(flattened_users) == 0 and len(tuserlist)!=0 and channel_busy==1:
                        print("All the remaining {} Wifi users are waiting".format(len(tuserlist)))
                        # pass
                        # do not break
                    if len(flattened_users)==0 and len(tuserlist)==0 and RTSuserlist ==0:
                        print("All wifi users have finished transmitting and are not programmed to do it again in this simulation")
                        break   # break here
                    
                    if CTS != 0:
                        
                        # Manage random backoff slots and DIFS as before
                        for ch in range(6):
                            #print(f"CTS active, processing tuserlist.")
                            #print("tuserlist ",[(u.ueID) for u in tuserlist[ch]])
                            #print("current status of random backoff ", [(u.ueID,u.random_backoff_slots) for u in tuserlist[ch] if u.random_backoff_flag==1])
                            #print("current status of DIFS ", [(u.ueID,u.DIFS_slots) for u in tuserlist[ch] if u.DIFS_flag==1])

                            for u in tuserlist[ch]:
                                if u.random_backoff_flag == 1 and u.random_backoff_slots > 0 and u.DIFS_flag == 0:
                                    u.random_backoff_slots -= 1

                                if u.random_backoff_flag == 1 and u.random_backoff_slots == 0 and u.DIFS_flag == 0:
                                    u.setRandomBackoff()
                        
                        if channel_busy[selected_user.channel] == 1:
                            #print(f"Wifi user {selected_user.ueID} in channel {selected_user.channel} used 1 slot during LTE's period.")
                            WifiCountU += 1
                        else:
                            selected_user.req_no_wifi_slot -= 1
                            WifiCountS += 1
                            selected_user.bits_sent += scene_params.get_bits_per_wifi_slot_from_Mbps(
                                selected_user.bs.bits_per_symbol_of_user[bringRealUser(selected_user, wifi_users_list)]
                            )
                            total_Wifi_bits_sent += scene_params.get_bits_per_wifi_slot_from_Mbps(
                            selected_user.bs.bits_per_symbol_of_user[bringRealUser(selected_user, wifi_users_list)]
                            )

                            #print(Wifisensecount," Success ",[(u.ueID,u.DIFS_slots) for u in tuserlist[u.channel]])
                            #print(" Wifi user ",selected_user.ueID," used 1 slot successfully")

                        CTS -= 1

                        # Handle cases when CTS reaches zero
                        if CTS == 0:
                            if channel_busy[selected_user.channel] == 1:
                                #print("User ",selected_user.ueID, "was till now sending during period 0 and is added back to allwuss")
                                #print("\n")
                                allwuss[selected_user.channel].append(selected_user)
                            elif channel_busy[selected_user.channel] == 0 and selected_user.req_no_wifi_slot > 0:
                                allwuss[selected_user.channel].append(selected_user)
                            elif channel_busy[selected_user.channel] == 0 and selected_user.req_no_wifi_slot == 0:
                                selected_user.req_no_wifi_slot = (selected_user.req_data_rate*10)/(selected_user.bs.bits_per_symbol_of_user[bringRealUser(selected_user, wifi_users_list)]*9)
                                selected_user.req_no_wifi_slot = int(math.ceil(selected_user.req_no_wifi_slot))
                                FinishedWifilist.append(selected_user)
                                #print("User ",selected_user.ueID, "has completed his transmission compleetly and is added back to allwuss")
                            continue

                    if CTS == 0:
                        
                        #print("current status of random backoff", [(u.ueID,u.random_backoff_slots) for u in flattened_users if u.random_backoff_flag==1])
                        #print("current status of DIFS", [(u.ueID,u.DIFS_slots) for u in flattened_users if u.DIFS_flag==1])
                        ready_users = []
                        for i in range(6):
                        # Pass the flattened list to assignProb2
                            service.assignProb2(allwuss[i])
                            #print([u.probability for u in allwuss[i]])
                            Wifiuserscount, ready_user = service.countWifiUsersWhoTransmit(allwuss[i])
                            ready_users.append(ready_user)
                        
                        
                        for ch in range(6):
                            #print("New Users who want to transmit: ",[x.ueID for x in ready_users[ch]])
                            if channel_busy[ch] == 1:
                                #print("current status of random backoff", [(u.ueID, u.random_backoff_slots) for u in ready_users[ch] if u.random_backoff_flag == 1])
                                WifiCountU += 1
                                
                                # For users in the current channel
                                for u in ready_users[ch]:
                                    if u.random_backoff_flag == 0 and u.random_backoff_slots == 0 and u.DIFS_flag == 0:
                                        u.random_backoff_flag = 1
                                        u.setRandomBackoff()

                                # Handling random backoff for tuserlist users in the current channel
                                for u in tuserlist[ch]:
                                    if u.random_backoff_flag == 1 and u.random_backoff_slots > 0 and u.DIFS_flag == 0:
                                        u.random_backoff_slots -= 1
                                    if u.random_backoff_flag == 1 and u.random_backoff_slots == 0 and u.DIFS_flag == 0:
                                        u.random_backoff_flag = 1
                                        u.setRandomBackoff()
                                    if u.random_backoff_flag == 0 and u.random_backoff_slots == 0 and u.DIFS_flag == 0:
                                        u.random_backoff_flag = 1
                                        u.setRandomBackoff()

                                # Adding users to tuserlist for the current channel
                                for u in ready_users[ch]:
                                    tuserlist[ch].append(u)
                                    #print(f"Channel {u.channel} allwuss list before removal: {[user.ueID for user in allwuss[u.channel]]}")

                                    if u in allwuss[u.channel]:
                                        allwuss[u.channel].remove(u)
                                    else:
                                        print(f"Warning: User {u.ueID} not found in channel {u.channel}.")

                            if channel_busy[ch] == 0:
                                #print("Entered that channel is free")
                                # DIFS and RTS handling per channel
                                remove_from_tuserlist_RTS = [[],[],[],[],[],[]]
                                for u in tuserlist[ch]:
                                    if u.random_backoff_flag == 1 and u.random_backoff_slots > 0 and u.DIFS_flag == 0:
                                        u.random_backoff_slots -= 1
                                    elif u.random_backoff_flag == 1 and u.random_backoff_slots == 0 and u.DIFS_flag == 0:
                                        u.random_backoff_flag = 0
                                        u.DIFS_flag = 1
                                        u.DIFS_slots = scene_params.DIFS_slots
                                    if u.random_backoff_flag == 0 and u.DIFS_flag == 1:
                                        if u.DIFS_slots > 0:
                                            u.DIFS_slots -= 1
                                        if u.DIFS_slots == 0:
                                            u.DIFS_flag = 0
                                            u.DIFS_slots = scene_params.DIFS_slots
                                            RTSuserlist[ch].append(u)
                                            #print("Appended in RTS", u.ueID)
                                            remove_from_tuserlist_RTS[ch].append(u)

                                # Remove users from tuserlist after RTS processing
                                for u in remove_from_tuserlist_RTS[ch]:
                                    tuserlist[ch].remove(u)

                                if len(RTSuserlist[ch]) > 0:
                                    selected_user = service.sendRTS(scene_params, RTSuserlist[ch])
                                    selected_user.RTS_flag = 1
                                    RTSuserlist[ch].remove(selected_user)

                                    #print(f"Selected userid: {selected_user.ueID}")
                                    CTS = selected_user.req_no_wifi_slot
                                    t_req_no_wifi_slot = selected_user.req_no_wifi_slot
                                else:
                                    WifiCountU += 1

                            
                    Wifisensecount += 1
                    rem_wifi_slots -= 1
                    #end of the while loop
                #print("\nWifi Successful: ",WifiCountS," Wifi Unused: ",WifiCountU,"\n")

                for antenna_index in range(3):
                    # More than one LTE BS has zero
                    if multiple_zero == 1:
                        LTECountU[antenna_index]+=4
                        continue

                    if single_zero[antenna_index] == 1:
                        LTEsubframeS = 0
                        half_ms = 2

                        while half_ms:
                            LTE_proportions = []
                            selected_bs = lte_bs_list[lbs_single_transmission_ind[antenna_index]]
                            LTE_proportions = service.calculate_LTE_proportions(scene_params, selected_bs.user_list_antenna[antenna_index])

                            #print(f"Antenna {antenna_index} LTE Proportions:", LTE_proportions)

                            give = 0
                            for u in selected_bs.user_list_antenna[antenna_index]:
                                if u.transmission_finished == 1:
                                    continue

                                #print(u.req_no_PRB, LTE_proportions[give])

                                if u.req_no_PRB <= LTE_proportions[give]:
                                    givenPRB = u.req_no_PRB
                                    u.req_no_PRB = 0
                                    u.transmission_finished = 1
                                    #print("User", u.ueID, "has finished transmission")
                                    u.bits_sent += givenPRB * scene_params.PRB_total_symbols * u.LTE_bs.bits_per_symbol_of_user[u]
                                    total_LTE_bits_sent += givenPRB * scene_params.PRB_total_symbols * u.LTE_bs.bits_per_symbol_of_user[u]

                                    LTECountS[antenna_index] += givenPRB
                                    LTEsubframeS += givenPRB

                                    service.calculate_LTE_user_PRB(scene_params, [u])

                                    if u.req_no_PRB <= 0:
                                        u.req_no_PRB = 1
                                else:
                                    u.req_no_PRB -= LTE_proportions[give]
                                    u.bits_sent += LTE_proportions[give] * scene_params.PRB_total_symbols * u.LTE_bs.bits_per_symbol_of_user[u]
                                    total_LTE_bits_sent += LTE_proportions[give] * scene_params.PRB_total_symbols * u.LTE_bs.bits_per_symbol_of_user[u]

                                    LTECountS[antenna_index] += LTE_proportions[give]
                                    LTEsubframeS += LTE_proportions[give]

                                give += 1

                            total_PRBs[antenna_index] += 100
                            half_ms -= 1

                            #print(f"Successful RB allocation for antenna {antenna_index} till now: ", LTECountS)

                        scene_params.pTx_one_PRB[antenna_index] = PARAMS.pTxLTE[antenna_index]/(2*scene_params.PRB_total_prbs*scene_params.duration_frame*100)

                        LTEPowerS += LTEsubframeS * scene_params.pTx_one_PRB[antenna_index]

                        #print(f"Power consumed by antenna {antenna_index} this subframe: ", LTEsubframeS * scene_params.pTx_one_PRB[antenna_index])

                if subframe_iterator == 9:

                    # if thisparams.vary_load == 0:
                    CTS = 0

                    for u in FinishedWifilist:
                        allwuss[u.channel].append(u)

                    for channel_users in tuserlist:
                        for u in channel_users:
                            # Append each user `u` to the appropriate channel list in `allwuss`
                            allwuss[u.channel].append(u)

                    for channel_users in RTSuserlist:
                        for u in channel_users:
                            # Append each user `u` to the appropriate channel list in `allwuss`
                            allwuss[u.channel].append(u)

                    flattened_allwuss = [user for sublist in allwuss for user in sublist]
                    flattened_RTS = [user for sublist in RTSuserlist for user in sublist]
                    flattened_tuserlist = [user for sublist in tuserlist for user in sublist]
                    is_selected_user_present = 0
                    for selected_user in chain(flattened_RTS, flattened_tuserlist, FinishedWifilist):
                        for u in flattened_allwuss:
                            if selected_user.ueID == u.ueID:
                                is_selected_user_present = 1
                                break
                        
                        if is_selected_user_present == 1:
                            pass
                        else:
                            selected_user.req_no_wifi_slot = (selected_user.req_data_rate*10)/(selected_user.bs.bits_per_symbol_of_user[bringRealUser(selected_user, wuss)]*9)
                            selected_user.req_no_wifi_slot = int(math.ceil(selected_user.req_no_wifi_slot))
                            allwuss[selected_user.channel].append(selected_user)
                    for ch in range(6):
                        for u in allwuss[ch]:
                            u.DIFS_flag = 0
                            u.DIFS_slots = scene_params.DIFS_slots
                            u.random_backoff_flag = 0
                            u.random_backoff_slots = 0

                    FinishedWifilist = []
                    tuserlist = [[],[],[],[],[],[]]
                    RTSuserlist = [[],[],[],[],[],[]]
                    
                    for b in lte_bs_list:
                        for u in b.user_list:
                            u.transmission_finished = 0
                            service.calculate_LTE_user_PRB(scene_params,[u])

                    ###### HERE, Varying of Users starts
                    if scene_params.vary_load == 1 and vary_for_every <=0 :
                        
                        # LTE_vary_factor = service.Vary_Load(thisparams, LTE_vary_factor)
                        # Wifi_vary_factor = service.Vary_Load(thisparams, Wifi_vary_factor)

                        # Caluclate new count of users
                        # newLTEuserscount = math.ceil(LTE_vary_factor*thisparams.numofLTEUE)
                        # newWifiuserscount = math.ceil(Wifi_vary_factor*thisparams.numofWifiUE)
                        CTS = 0

                        newLTEuserscount = scene_params.set_users_LTE[scene_params.vary_iterator]
                        newWifiuserscount = scene_params.set_users_Wifi[scene_params.vary_iterator]

                        scene_params.vary_iterator += 1

                        # Clear older lists
                        for lb in lte_bs_list:
                            lb.user_list = np.array([])
                            lb.t_user_list = np.array([])
                            lb.user_list_antenna = [np.array([]), np.array([]), np.array([])]  # Separate user lists for each antenna
                            lb.bits_per_symbol_of_user = dict()
                        
                        for wb in wifi_bs_list:
                            wb.user_list = np.array([])  
                            wb.t_user_list = np.array([])
                            wb.bits_per_symbol_of_user = dict()

                        allwuss = [[],[],[],[],[],[]]
                        tuserlist = [[],[],[],[],[],[]]
                        RTSuserlist = [[],[],[],[],[],[]]
                        FinishedWifilist = [[],[],[],[],[],[]]

                        # Create new users
                        varyparams = PARAMS()
                        varyparams.numofLTEUE = newLTEuserscount
                        varyparams.numofWifiUE = newWifiuserscount

                        LTE_users_list = service.createLTEUsers(varyparams,lte_bs_list[0],wifi_bs_list[0])
                        wifi_users_list = service.createWifiUsers(varyparams,lte_bs_list[0],wifi_bs_list[0])

                        # Connecting all the LTE UE with a LTE BS
                        i = 0
                        for u in LTE_users_list:
                            ind = u.measureSINRandConnect(lte_bs_list,wifi_bs_list)

                            # if ind is -1 then that user is out of range of any BS
                            '''if ind == -1:
                                luss = np.delete(luss,i)
                                continue'''

                            # Add this UE to user_list
                            lte_bs_list[ind].user_list = np.append(lte_bs_list[ind].user_list, u)
                            i+=1


                        # Keeping a copy of LTE transmitting users
                        for b in lte_bs_list:
                            for element in b.user_list:
                                b.t_user_list = np.append(b.t_user_list,element)

                            b.lusscount = len(b.t_user_list)

                        # Connecting all the Wifi UE with a Wifi BS
                        i = 0
                        for u in wifi_users_list:
                            ind = u.measureSNRandConnect(lte_bs_list,wifi_bs_list)
                            # if ind is -1 then that user is out of range of any BS
                            '''if ind == -1:
                                wuss = np.delete(wuss,i)
                                continue'''

                            # Add this UE to user_list
                            wifi_bs_list[ind].user_list = np.append(wifi_bs_list[ind].user_list, u)
                            i+=1

                        # Keeping a copy of Wifi transmitting users
                        for b in wifi_bs_list:
                            for element in b.user_list:
                                b.t_user_list = np.append(b.t_user_list,element)

                            b.wusscount = len(b.t_user_list)

                        # Based on ratios decided by the user, assign data rates to UE
                        service.assign_data_rate_to_users(scene_params, LTE_users_list, wifi_users_list)

                        SINR=[]
                        SNR=[]

                        # Measuring SINR for LTE Users
                        for u in LTE_users_list:
                            SINR.append(u.SINR)

                        service.decide_LTE_bits_per_symbol(lte_bs_list,scene_params)
                        service.calculate_LTE_user_PRB(scene_params, LTE_users_list)

                        for u in wifi_users_list:
                            SNR.append(u.SNR)

                        service.decide_wifi_bits_per_symbol(wifi_bs_list, scene_params)
                        service.calculate_wifi_user_slots(scene_params, wifi_users_list)

                        for b in wifi_bs_list:
                            b.t_user_list = b.user_list
                        
                        allwuss = [[], [], [], [], [], []]  # List of all waiting users per channel
                        for user in wifi_users_list:
                            if user.best_lte_region == "0":
                                user.channel = 0
                                allwuss[0].append(copy.deepcopy(user))
                            elif user.best_lte_region == "1":
                                user.channel = 1
                                allwuss[1].append(copy.deepcopy(user))
                            elif user.best_lte_region == "2":
                                user.channel = 2
                                allwuss[2].append(copy.deepcopy(user))
                            elif user.best_lte_region == "01":
                                user.channel = 3
                                allwuss[3].append(copy.deepcopy(user))
                            elif user.best_lte_region == "12":
                                user.channel = 4
                                allwuss[4].append(copy.deepcopy(user))
                            elif user.best_lte_region == "20":
                                user.channel = 5
                                allwuss[5].append(copy.copy(user))

                        #print("LTE users {} at iteration {}".format(varyparams.numofLTEUE,tf))
                        #print("Wifi users {} at iteration {}".format(varyparams.numofWifiUE,tf))


                        vary_for_every = scene_params.vary_for_every
                        #End of subframe loop
            U_LTE = 0
            U_Wifi = 0
            for i in range(3):
                U_LTE += (LTECountS[i])/(total_PRBs[i]) if (total_PRBs[i]) != 0 else 0
            U_LTE /= 3
            U_Wifi = (WifiCountS)/total_Wifi_slots if total_Wifi_slots != 0 else 0
            frame_fairness = ((U_LTE+U_Wifi)**2)/(2*((U_LTE**2)+(U_Wifi**2)))

            Fairness.append(frame_fairness)
            print("Fairness:", frame_fairness)
            reward = rl.RewardFunction(frame_fairness,scene_params)
            
            rl.remember(rl.previous_state, rl.current_action, reward, rl.current_state, False)
            replay_interval = max(1, int(1000 / len(rl.memory)))
            if tf % 2000 == 0:
                rl.replay()
            if tf % 30000 == 0:
                rl.update_target_network()   
            rl.reward_history.append(reward)
            if tf >= rl.window_size:
                    if rl.check_convergence(rl.reward_history):
                        consecutive_success += 1
                        if consecutive_success >= rl.patience:
                            print("Converged!")
                            break
                    else:
                        consecutive_success = 0
            frame_T_LTE = (total_LTE_bits_sent * 10**3)/scene_params.duration_frame
            
            LTE_Throughput.append(frame_T_LTE)
            #print("LTE_Throughput: ", frame_T_LTE)
            LTE_Power.append(LTEPowerS)
            #print("LTE_Power: ",LTEPowerS)
            scene_params.pTxLTE = rl.originalPower
            ECR.append((100*LTEPowerS)/frame_T_LTE)
            #print("ECR: ", (100*LTEPowerS)/frame_T_LTE)
            Utilization.append(U_LTE)
            #print("LTE Utilization: ",U_LTE)
            Wifi_Utilization.append(U_Wifi)
            #print("Wifi Utilization: ",U_Wifi)
            
            LTE_user_satisfy = 0
            for i in range(3):
                LTE_user_satisfy += LTECountS[i]/(3*ltereq[i])
            LTE_User_satisfy.append(LTE_user_satisfy)
            #print("LTE_User_satisfy: ", LTE_user_satisfy)
            # print(LTECountS)
            Wifi_User_satisfy.append(WifiCountS/wifi_req)
            #print("Wifi_User_satisfy: ", WifiCountS/wifi_req)
            # print(WifiCountS)
            frame_T_Wifi = (total_Wifi_bits_sent * 10**3)/scene_params.duration_frame
            Wifi_Throughput.append(frame_T_Wifi)
            #print("Wifi_Throughput: ", frame_T_Wifi)
        #rl.save("trained_model.weights.h5")
        
        F.append(max(Fairness))
        
        LTE_T.append(max(LTE_Throughput))
        
        LTE_P.append(min(LTE_Power))
        
        E.append(min(ECR))
        
        U.append(max(Utilization))
        
        Uw.append(max(Wifi_Utilization))
        
        LUS.append(max(LTE_User_satisfy))
        
        WUS.append(max(Wifi_User_satisfy))
        
        W_T.append(max(Wifi_Throughput))
    
    print("Fairness: ", sum(F)/100)
    print("LTE_Throughput: ", sum(LTE_T)/100)
    print("LTE_Power: ",sum(LTE_P)/100)
    print("ECR: ", sum(E)/100)
    print("LTE Utilization: ",sum(U)/100)
    print("Wifi Utilization: ",sum(Uw)/100)
    print("LTE_User_satisfy: ", sum(LUS)/100)
    print("Wifi_User_satisfy: ", sum(WUS)/100)
    print("Wifi_Throughput: ", sum(W_T)/100)
           
if __name__ == "__main__":
    main()