import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Params.ConstantParams import PARAMS
from entities.BaseStation import LTEBaseStation
from entities.BaseStation import WifiBaseStation
from entities.UserEquipment import UserEquipment
from entities.UserEquipment import WifiUserEquipment

class ServiceClass:

    # Returns List of Base Stations of size PARAMS.numofLTEBS
    # each BS with a sequential ID and random location in (length,breadth)
    # Locations are assigned based on scenes
    def createLTEBaseStations(self,scene_params,scenenum=1):
        
        bss = np.array([])

        if(scenenum == 1):
            b = LTEBaseStation()

            b.bsID = 0
            b.x = np.random.randint(40,60,scene_params.numofLTEBS)
            b.y = np.random.randint(40,60,scene_params.numofLTEBS)
            b.pTx = PARAMS().pTxLTE  # Watts

            bss = np.append(bss,b)

            return bss

        elif(scenenum == 2):
            b = LTEBaseStation()

            b.bsID = 0
            b.x = 35
            b.y = 50
            b.pTx = PARAMS().pTxLTE  # Watts

            bss = np.append(bss,b)

            return bss

        elif(scenenum == 3):
            # scene_params = PARAMS()

            scene_params.numofLTEBS = 3

            nums1 = np.random.randint(30,70,scene_params.numofLTEBS)
            nums2 = np.random.randint(30,70,scene_params.numofLTEBS)
            
            for i in range(0,scene_params.numofLTEBS):

                b = LTEBaseStation()

                b.bsID = i
                b.x = nums1[i]
                b.y = nums2[i]
                b.pTx = scene_params.pTxLTE  # Watts

                bss = np.append(bss,b)

            return bss

        elif(scenenum == 4):

            nums1 = np.random.randint(40,60,1)
            nums2 = np.random.randint(40,60,1)

            b = LTEBaseStation()

            b.bsID = 0
            b.x = nums1[0]
            b.y = nums2[0]
            b.pTx = PARAMS().pTxLTE  # Watts

            bss = np.append(bss,b)

            return bss

        elif(scenenum == 5):
            # scene_params = PARAMS()

            scene_params.numofLTEBS = 3

            nums1 = np.random.randint(30,70,scene_params.numofLTEBS)
            nums2 = np.random.randint(30,70,scene_params.numofLTEBS)
            
            for i in range(0,scene_params.numofLTEBS):

                b = LTEBaseStation()

                b.bsID = i
                b.x = nums1[i]
                b.y = nums2[i]
                b.pTx = scene_params.pTxLTE  # Watts

                bss = np.append(bss,b)

            return bss

        
        #------------------------- Generic Scene ----------------
        elif(scenenum == 0):
            nums1 = np.random.randint(1,PARAMS().length,scene_params.numofLTEBS)
            nums2 = np.random.randint(1,PARAMS().breadth,scene_params.numofLTEBS)

            for i in range(0,scene_params.numofLTEBS):

                b = LTEBaseStation()

                b.bsID = i
                b.x = nums1[i]
                b.y = nums2[i]
                b.pTx = PARAMS().pTxLTE  # Watts

                bss = np.append(bss,b)

            return bss
        
        else:
            print("Please choose scene specified in documentation")
            exit()

    # Returns List of Base Stations of size PARAMS.numofWifiBS
    # each BS with a sequential ID and random location in (length,breadth)
    # If there is one BS then assign static location
    def createWifiBaseStations(self,scene_params,scenenum=1):
        
        bss = np.array([])

        if(scenenum==1):
            b = WifiBaseStation()

            b.bsID = 0
            b.x = np.random.randint(30,70,scene_params.numofWifiBS)
            b.y = np.random.randint(30,70,scene_params.numofWifiBS)

            b.pTx = PARAMS().pTxWifi  # Watts

            bss = np.append(bss,b)

            return bss

        elif(scenenum==2):
            b = WifiBaseStation()

            b.bsID = 0
            b.x = 65
            b.y = 50
            b.pTx = PARAMS().pTxWifi  # Watts

            bss = np.append(bss,b)

            return bss

        elif (scenenum==3):
            
            # scene_params = PARAMS()

            scene_params.numofWifiBS = 3

            nums1 = np.random.randint(30,70,scene_params.numofWifiBS)
            nums2 = np.random.randint(30,70,scene_params.numofWifiBS)

            for i in range(0,scene_params.numofWifiBS):

                b = WifiBaseStation()

                b.bsID = i
                b.x = nums1[i]
                b.y = nums2[i]
                b.pTx = scene_params.pTxWifi  # Watts

                bss = np.append(bss,b)

            return bss

        elif (scenenum==4):
            
            # scene_params = PARAMS()

            scene_params.numofWifiBS = 3

            nums1 = np.random.randint(30,70,scene_params.numofWifiBS)
            nums2 = np.random.randint(30,70,scene_params.numofWifiBS)

            for i in range(0,scene_params.numofWifiBS):

                b = WifiBaseStation()

                b.bsID = i
                b.x = nums1[i]
                b.y = nums2[i]
                b.pTx = scene_params.pTxWifi  # Watts

                bss = np.append(bss,b)

            return bss

        elif(scenenum==5):
            b = WifiBaseStation()

            num1 = np.random.randint(40,60,1)
            num2 = np.random.randint(40,60,1)

            b.bsID = 0
            b.x = num1[0]
            b.y = num2[0]
            b.pTx = PARAMS().pTxWifi  # Watts

            bss = np.append(bss,b)

            return bss

        
        
        #-----------------------Generic Scene----------------------------
        elif (scenenum==0):
            nums1 = np.random.randint(1,PARAMS().length,PARAMS().numofWifiBS)
            nums2 = np.random.randint(1,PARAMS().breadth,PARAMS().numofWifiBS)

            for i in range(0,PARAMS.numofWifiBS):

                b = WifiBaseStation()

                b.bsID = i
                b.x = nums1[i]
                b.y = nums2[i]
                b.pTx = PARAMS().pTxWifi  # Watts

                bss = np.append(bss,b)

            return bss

        else:
            print("Please choose scene specified in documentation")
            exit()
    
    
    # Returns List of User Equipments of size PARAMS.numofLTEUE
    # each UE with a sequential ID and random location in (length,breadth)
    def createLTEUsers(self,scene_params, lte_bs, wifi_bs):
        #np.random.seed(scene_params.seed_valueLTE)
        uss = np.array([])
        nums1 = np.random.randint(30,70,scene_params.numofLTEUE)
        nums2 = np.random.randint(30,70,scene_params.numofLTEUE)

        for i in range(0,scene_params.numofLTEUE):

            u = UserEquipment()

            u.ueID = i
            
            if nums1[i] == lte_bs.x and nums2[i] == lte_bs.y :
                u.x = nums1[i]
                u.y = nums2[i]+5

            elif nums1[i] == wifi_bs.x and nums2[i] == wifi_bs.y :
                u.x = nums1[i]
                u.y = nums2[i]+5

            else:
                u.x = nums1[i]
                u.y = nums2[i]

            uss = np.append(uss,u)
        
        return uss

    # Returns List of User Equipments of size PARAMS.numofWifiUE
    # each UE with a sequential ID and random location in (length,breadth)
    def createWifiUsers(self,scene_params,lte_bs,wifi_bs):
        #np.random.seed(scene_params.seed_valueWifi)
        uss = np.array([])
        nums1 = np.random.randint(45,105,scene_params.numofWifiUE)
        nums2 = np.random.randint(20,80,scene_params.numofWifiUE)

        for i in range(0,scene_params.numofWifiUE):

            u = WifiUserEquipment()

            u.ueID = i

            if nums1[i] == lte_bs.x and nums2[i] == lte_bs.y :
                u.x = nums1[i]
                u.y = nums2[i]+5

            elif nums1[i] == wifi_bs.x and nums2[i] == wifi_bs.y :
                u.x = nums1[i]
                u.y = nums2[i]+5

            else:
                u.x = nums1[i]
                u.y = nums2[i]
            

            uss = np.append(uss,u)
        
        return uss
    
    
    def countWifiUsersWhoTransmit(self,wuss):
        WifiUsersWhoTransmit=0

        # ~Changed~
        # userlist contains list of lists
        # every row indicates the index of wifi BS
        # and each row has a list of users connected to that BS willing to transmit
        # ~~~~~~~~~~~~~~~~~~~~
        # List of users who want to trasmit
        
        userlist = []
        for u in wuss:
            if u.probability<PARAMS.prob:
                # temp.append(u)
                userlist.append(u)
                WifiUsersWhoTransmit += 1
        
        return WifiUsersWhoTransmit,userlist

    def assignProb(self,wbss):
        for b in wbss:
            for u in b.user_list:
                u.probability=round(random.uniform(0,1),4) #Assigning random number to each user 

    def assignProb2(self,wuss):
        for u in wuss:
            u.probability=round(random.uniform(0,1),4)
    
    # Calling this function will fill the values of Cumulative Probablities of Profiles in lists of LTE_profile_c_prob and Wifi_profile_c_prob in Constant Params
    def calculate_profile_prob(self,scene_params):

        # Manually calculating Cumulative Probabilities

        # STEP 1
        # Get sum of numbers
        total_sum=sum(scene_params.LTE_ratios)

        # ratio/sumofratios will give probability for that ratio
        for i in scene_params.LTE_ratios:
            scene_params.LTE_profile_prob.append(round((i/total_sum),1) ) 
        
        # -- Cumulative Prob --

        # First CumProb  = First Prob
        scene_params.LTE_profile_c_prob.append(scene_params.LTE_profile_prob[0])
        # Add further CumProb
        for i in range(1,len(scene_params.LTE_ratios)):
            scene_params.LTE_profile_c_prob.append(round((scene_params.LTE_profile_prob[i]+scene_params.LTE_profile_c_prob[i-1]),1))
        # Now we will get the CumProb in LTE_profile_c_prob in constant params

        # Repeat from STEP 1 for wifi
        total_sum=sum(scene_params.wifi_ratios)
        for i in scene_params.wifi_ratios:
            scene_params.wifi_profile_prob.append(round((i/total_sum),1))
        
        scene_params.wifi_profile_c_prob.append(scene_params.wifi_profile_prob[0])
        for i in range(1,len(scene_params.wifi_ratios)):
            scene_params.wifi_profile_c_prob.append(round((scene_params.wifi_profile_prob[i]+scene_params.wifi_profile_c_prob[i-1]),1))
    
    # 1. Assign Data Rate (Profile) to users based on probabilities and ratios
    # 2. This function will set the value 'req_data_rate' in LTE and Wifi UE
    # 3. 'service.calculate_profile_prob' must be called before this function
    def assign_data_rate_to_users(self,scene_params,luss,wuss):
        # LTE users
        #random.seed(scene_params.seed_valueLTE)
        for u in luss:
            # get a probability value between 0-1
            temp_prob=round(random.random(),2)
            
            # Based on temp_prob, assign datarate
            k=0
            for i in scene_params.LTE_profile_c_prob:
                if(temp_prob<=i):
                    u.req_data_rate= scene_params.LTEprofiles[k]
                    break
                k+=1
        
        # Wifi users
        #random.seed(scene_params.seed_valueWifi)
        for u in wuss:
            # get a probability value between 0-1
            temp_prob2=round(random.random(),2)

            # Based on temp_prob, assign datarate
            k=0
            for i in scene_params.wifi_profile_c_prob:
                if(temp_prob2<=i):
                    u.req_data_rate=scene_params.Wifiprofiles[k]
                    break
                k+=1

        # This function returns the value 'bits per symbol' for corresponding SINR value
    # from the LTE_MCS dictionary in constant_params
    def get_LTE_bits_per_symbol(self,sinr,scene_params):

        given_sinr = list(scene_params.LTE_MCS.keys())

        i = 0
        for x in given_sinr:

            if sinr<=x:
                # if SINR is equal to first value in MCS
                if i==0:
                    return scene_params.LTE_MCS[given_sinr[0]]
                else:
                    return scene_params.LTE_MCS[given_sinr[i-1]]
            i+=1
        
        if sinr >= given_sinr[i-1]:
            return scene_params.LTE_MCS[given_sinr[i-1]]


    # This function fills the 'BS.bits_per_symbol_of_user'
    # 'BS.bits_per_symbol_of_user' 
    def decide_LTE_bits_per_symbol(self,lbss,scene_params):
        
        # For each LTE Base Station
        for b in lbss:
            # For every user connected to BS b
            for u in b.user_list:
                # get the bits per symbol
                bos = self.get_LTE_bits_per_symbol(u.SINR,scene_params)
                # store bits per symbol in 'BS.bits_per_symbol_of_user'
                # with user as key and value as its bits per symbol
                b.bits_per_symbol_of_user[u] = bos
    
    # This function calculates total no of PRB required by each user
    #  and stores it in 'UE.req_no_PRB'
    def calculate_LTE_user_PRB(self,scene_params,luss):

        for u in luss:
            # total PRB required by user = (required bits per slot)
            #                           /(bits per symbol)*(total symbols in PRB)
            # u.req_no_PRB = scene_params.get_bits_per_slot_from_Kbps(u.req_data_rate)/(u.bs.bits_per_symbol_of_user[u]*scene_params.PRB_total_symbols)
            u.req_no_PRB = (u.req_data_rate*(10**3)*10*(10**-3))/(u.LTE_bs.bits_per_symbol_of_user[u]*scene_params.PRB_total_symbols)
            u.req_no_PRB = math.ceil(u.req_no_PRB)
            
    def getTotalRequiredPRB(self, scene_params, luss):
        ##THIS WORKS FOR 1 BASE STATION
        # Dictionary to store total PRBs required per antenna
        prbs_per_antenna = {}

        # Iterate over each user in the list of users
        for u in luss:
            antenna_id = u.best_antenna_index  # Assuming each user has an attribute 'antenna_id'
            
            # Initialize the PRB count for this antenna if not already in the dictionary
            if antenna_id not in prbs_per_antenna:
                prbs_per_antenna[antenna_id] = 0
            
            # Accumulate PRBs needed by this user for the specific antenna
            prbs_per_antenna[antenna_id] += u.req_no_PRB

        return prbs_per_antenna


    def get_wifi_bits_per_symbol(self,snr,scene_params):

        given_snr = list(scene_params.wifi_MCS.keys())

        i = 0
        for x in given_snr:
            if snr<=x:
                # if SNR is equal to first value in MCS
                if i==0:
                    return scene_params.wifi_MCS[given_snr[0]]
                else:
                    return scene_params.wifi_MCS[given_snr[i-1]]
            i+=1
        
        if snr >= given_snr[-1]:
            return scene_params.wifi_MCS[given_snr[-1]]

    # This function fills the 'BS.bits_per_symbol_of_user'
    # 'BS.bits_per_symbol_of_user' 
    def decide_wifi_bits_per_symbol(self,wbss,scene_params):
        
        # For each Wifi Base Station
        for b in wbss:
            # For every user connected to BS b
            for u in b.user_list:
                # get the bits per symbol
                bos = self.get_wifi_bits_per_symbol(u.SNR,scene_params)
                # store bits per symbol in 'BS.bits_per_symbol_of_user'
                # with user as key and value as its bits per symbol
                b.bits_per_symbol_of_user[u] = bos

    # This function calculates total no of wifi slots required by each user
    #  and stores it in 'UE.req_no_wifi_slot'
    def calculate_wifi_user_slots(self,scene_params,wuss):

        for u in wuss:
            # total wifi slots required by user = (required datarate)
    #                                                               / (available datarate)
            u.req_no_wifi_slot = (u.req_data_rate*10)/(u.bs.bits_per_symbol_of_user[u]*9)
            u.req_no_wifi_slot = math.ceil(u.req_no_wifi_slot)

    

    def getTotalRequiredWifiSlot(self, scene_params, wuss):
        """Calculate total required WiFi slots for each region based on the user's antenna segment."""
        
        # Dictionary to store total WiFi slots required for each region
        region_slots = {
            "0": 0,
            "1": 0,
            "2": 0,
            "01": 0,
            "12": 0,
            "20": 0
        }
        total = 0
        # Sum up the required slots for each region based on `best_lte` label
        for u in wuss:
            total += u.req_no_wifi_slot
            if u.best_lte_region in region_slots:
                region_slots[u.best_lte_region] += u.req_no_wifi_slot
            else:
                print(f"Warning: Unknown region {u.best_lte_region} for user {u.ueID}")

        return region_slots,total

    # Creates CSVs of locations of BSs and Users
    def createLocationCSV(self, wbss, lbss, luss, wuss):
        wbssl = []
        lbssl = []
        wussl = []
        lussl = []

        # Creating CSVs
        for bs in wbss:
            wbssl.append((bs.x, bs.y))
        wbssdf = pd.DataFrame(wbssl)

        for bs in lbss:
            lbssl.append((bs.x, bs.y))
        lbssdf = pd.DataFrame(lbssl)

        for bs in wuss:
            wussl.append((bs.x, bs.y))
        wussdf = pd.DataFrame(wussl)

        for bs in luss:
            lussl.append((bs.x, bs.y))
        lussdf = pd.DataFrame(lussl)

        wussdf.to_csv("wussdf.csv", index=False)
        lussdf.to_csv("lussdf.csv", index=False)
        wbssdf.to_csv("wbssdf.csv", index=False)
        lbssdf.to_csv("lbssdf.csv", index=False)
        return
    
    def sendRTS(self,scene_params,RTSuserlist):

        selecteduser=random.choice(RTSuserlist)

        return selecteduser
    
    def calculate_LTE_proportions(self,scene_params,luss):
        LTE_proportions = []

        total = 0

        for u in luss:
            if u.transmission_finished == 0:
                total += u.req_no_PRB

        total2 = 0
        
        i = 0
        for u in luss:
            if u.transmission_finished == 1:
                continue

            k = math.ceil((u.req_no_PRB/total)*scene_params.PRB_total_prbs)
            # print(k," k")
            
            if total2 + k > 100:
                LTE_proportions.append(0)
                LTE_proportions[-1] = min(u.req_no_PRB,scene_params.PRB_total_prbs - sum(LTE_proportions[:-1]))
                
                for j in range(len(luss)-i-1):
                    LTE_proportions.append(0)

                break

            else:
                LTE_proportions.append(min(k,u.req_no_PRB))
                total2 += k
            
            i+=1

        # LTE_proportions[-1] = scene_params.PRB_total_prbs - sum(LTE_proportions[:-1])
        return LTE_proportions
    
class GraphService:

    def PlotScene(self,scenenum,description):

        wussdf = pd.read_csv("wussdf.csv")
        lussdf = pd.read_csv("lussdf.csv")
        wbssdf = pd.read_csv("wbssdf.csv")
        lbssdf = pd.read_csv("lbssdf.csv")

        x1 = lussdf.iloc[:, 0:1]
        y1 = lussdf.iloc[:, 1:2]

        x2 = wussdf.iloc[:, 0:1]
        y2 = wussdf.iloc[:, 1:2]

        x3 = lbssdf.iloc[:, 0:1]
        y3 = lbssdf.iloc[:, 1:2]

        x4 = wbssdf.iloc[:, 0:1]
        y4 = wbssdf.iloc[:, 1:2]

        plt.scatter(x1, y1, marker='x', color='red')
        plt.scatter(x2, y2, marker='x', color='blue')
        plt.scatter(x3, y3, marker='^', color='red', s=100)
        plt.scatter(x4, y4, marker='^', color='blue', s=100)
        plt.legend(["LTE User", "Wi-Fi User", "LTE BS", "Wi-Fi BS"],fontsize=14)
        plt.xlim(0, 100)
        plt.ylim(0, 100)

        plt.xlabel("X-Coordinates",fontsize=18)
        plt.ylabel("Y-Coordinates",fontsize=18)
        plt.title("Scene{} : {} LTE BS & {} Wi-Fi BS, {}".format(scenenum,len(x3),len(x4),description),fontsize=18)

        plt.show()

    def PlotHistSINR(self,SINR,scene_params):

        maxind=0
        minind=0
        maxSINR=max(SINR)
        minSINR=min(SINR)
        avg=maxSINR/5
        xt=[]
        here=minSINR
        for i in range(0,5):
            xt.append(round(here,2))
            here+=avg
    
        xt.append(round(maxSINR,2))

        lab = []
        mids = []

        for i in range(0,5):
            one = xt[i]
            two = xt[i+1]
            temp = one+two/2
            mids.append(temp)
            lab.append(str(one)+" to "+str(two))

        




        print("SINR:MIN = {}, MAX = {}".format(minSINR,maxSINR))
        # for i in range(1,len(SINR)):
        #     if(SINR[i]>maxSINR):
        #         maxSINR=SINR[i]
        #         maxind=i
        # if (SINR[i] < minSINR):
        #     minSINR = SINR[i]
        #     minind = i


        plt.hist(SINR,label="SINR of LTE Users", bins=5, edgecolor="black")
        plt.title("SINR of LTE Users",fontsize=18)
        plt.ylabel("No. of LTE Users",fontsize=18)
        plt.xlabel("SINR",fontsize=18)
        yt=range(1,scene_params.numofLTEUE+1)

        plt.yticks(yt,fontsize=14)
        # plt.xticks(mids,labels=lab,fontsize=14)
        plt.show()

    def PlotHistSNR(self,SNR,scene_params):

        maxind=0
        minind=0
        maxSNR=max(SNR)
        minSNR=min(SNR)
        
        avg=maxSNR/5
        xt=[]
        here=minSNR
        for i in range(0,5):
            xt.append(here)
            here+=avg
        xt.append(maxSNR)
        
        print("========")
        print(xt)

        print("SNR:MIN = {}, MAX = {}".format(minSNR,maxSNR))
        # for i in range(1,len(SINR)):
        #     if(SINR[i]>maxSINR):
        #         maxSINR=SINR[i]
        #         maxind=i
        # if (SINR[i] < minSINR):
        #     minSINR = SINR[i]
        #     minind = i

        plt.hist(SNR,label="SNR of Wi-Fi Users", bins=5, edgecolor="black")
        plt.title("SNR of Wi-Fi Users",fontsize=18)
        plt.ylabel("No. of Wi-Fi Users",fontsize=18)
        plt.xlabel("SNR",fontsize=18)
        yt=range(1,scene_params.numofWifiUE+1)
        plt.yticks(yt,fontsize=14)
        plt.xticks(xt,fontsize=14)

        plt.show()
