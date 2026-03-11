import numpy as np
from utils import HetNet_env, mobility_trace, collect_dataset

# parameters setting
room_env = {# 'Room1-2': {'AP_num': 6, 'room_length': 5, 'room_width': 5, 'UE_range': [5, 10, 15, 20]},
            # 'Room2-2': {'AP_num': 7, 'room_length': 6, 'room_width': 6, 'UE_range': [5, 10, 15, 20]},
            'Room8': {'AP_num': 7, 'room_length': 5, 'room_width': 8, 'UE_range': [10,15]}}
time_duration = 1 # seconds
time_slots = int(time_duration/0.1) # sample slots
room_list = ['Room8']
velocity = 2 # m/s
R_aver = 100 # average data rate requirement in Mbps
for room in room_list:
    room_env_now = room_env[room]
    Room_size = [room_env_now['room_length'], room_env_now['room_width']] # length and width of the room
    AP_num = room_env_now['AP_num'] # number of access points (1 WiFi, AP_num-1 LiFi APs)
    UE_range = room_env_now['UE_range']
    for trace_num in range(0,7): 
        UE_num = np.random.choice(UE_range) # number of user equipments
        # load env
        env = HetNet_env(AP_num=AP_num, UE_num=UE_num, X_length=Room_size[0], Y_length=Room_size[1], Z_height=3, room_mode=room)
        # generate mobility trace
        UE_traces = mobility_trace(UE_num, Room_size[0], Room_size[1], velocity, time_duration)
        # plot_trace(UE_traces, user_index=0)  # plot the trajectory of the first user
        ############ generate data rate requirement in each trail ############
        R_raw = np.random.gamma(1, R_aver, UE_num)
        env.R_requirement = np.clip(R_raw, 10, 500).tolist() # type: ignore
        for time_step in range(len(UE_traces[0])):
            UE_positions = [UE_traces[user_id][time_step] for user_id in range(UE_num)]
            # Task 1: update CSI information based on UE positions
            env.update_CSI(UE_positions = UE_positions)
            # Task 2: access point selection based on GT
            GT_results = env.load_balancing_GT(RA_mode=1)
            # Task 3: resource allocation using JRA algorithm
            env.Rho_iu = env.JRA() # type: ignore # AP_num * UE_num
            # collect dataset for three tasks
            json_file1="/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/%snew_task1_trace%s.json"%(room,trace_num+1)
            json_file2="/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/%snew_task2_trace%s.json"%(room,trace_num+1)
            json_file3="/data/LLM_indoor/LLaMA-Factory-main/Dataset-collection/dataset/dataset3/%snew_task3_trace%s.json"%(room,trace_num+1)
            collect_dataset(env, UE_positions, time_step, json_file1, json_file2, json_file3)










