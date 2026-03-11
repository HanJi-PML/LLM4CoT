# Author: Han, 11 Nov 2025
import numpy as np
import math
from pymobility_master.src.pymobility.models.mobility import gauss_markov, random_walk, random_waypoint
import json
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

class HetNet_env():
    def __init__(self, AP_num, UE_num, X_length, Y_length, Z_height, room_mode):
        self.AP_num = AP_num
        self.UE_num = UE_num
        self.B_LiFi = 40  # unit: Mbps
        self.B_WiFi = 20  # unit: Mbps
        self.X_length = X_length
        self.Y_length = Y_length
        self.Z_height = Z_height
        self.AP_positions, self.interference_matrix = get_AP_positions(room_mode=room_mode) # input this manually

    def update_CSI(self, UE_positions):
        # calculate SINR matrix and Capacity matrix based on current UE positions
        self.SINR_matrix = []
        self.Capacity = []
        Signal_power_matrix = []
        Capacity_list = [0]*self.AP_num
        for i in range(self.UE_num):
            Signal_power_list = []
            for j in range(self.AP_num):
                if j == 0:
                    mode = 'WiFi'
                else:
                    mode = 'LiFi'
                user_position = UE_positions[i] + [0]
                Signal_power = Signal_power_calculation(self.X_length, self.Y_length, self.Z_height, self.AP_positions[j], user_position, mode) #
                Signal_power_list.append(Signal_power)
            Signal_power_matrix.append(Signal_power_list)
        SINR_matrix = SINR_calculation(self.AP_num, self.UE_num, Signal_power_matrix, self.interference_matrix) # AP_num * UE_num
        Capacity_list[0] = self.B_WiFi * np.log2(1 + np.array(SINR_matrix[0])) # WiFi capacity
        Capacity_list[1:] = self.B_LiFi/2 * np.log2(1 + (np.exp(1)/(2*np.pi))*np.array((SINR_matrix[1:]))) # LiFi capacity
        SINR_matrix = np.maximum(np.array(SINR_matrix), 0.01)
        SINR_matrix_dB = 10 * np.log10(SINR_matrix)
        self.SINR_matrix = SINR_matrix_dB # AP_num * UE_num
        self.Capacity = np.array(Capacity_list) # AP_num * UE_num

    def load_balancing_SSS(self):
        # do SSS LB only for target UE
        # for TCP only
        SINR_matrix = self.SINR_matrix.T.tolist()  # UE_num * AP_num
        X_iu = []
        for i in range(self.UE_num):
            SINR_list = SINR_matrix[i]
            max_SNR = max(SINR_list)
            index = SINR_list.index(max_SNR)
            X_iu.append(index + 1)
        self.X_iu = X_iu

    def load_balancing_GT(self, RA_mode=None):
        ####### for TCP only
        self.load_balancing_SSS() # use SSS as initial X_iu
        # calculate initial satisfaction list
        Rho_iu = RA_optimization(self.AP_num, self.UE_num, self.X_iu, self.R_requirement, self.Capacity, opt_mode=RA_mode)
        Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num
        Satisfaction_vector = []
        for ii in range(self.UE_num):
            list1 = self.Capacity[:,ii]
            list2 = Rho_transposed[ii]
            sat_now = min(sum(list(np.multiply(list1, list2)))/self.R_requirement[ii], 1)
            Satisfaction_vector.append(sat_now)
        #######
        N_f = np.minimum(self.AP_num, 5)  # number of AP candidates for each UE
        count = 0
        mode = 0
        payoff_vector = [0]
        N = self.UE_num
        X_iu_old = self.X_iu

        while mode <= N:
            estimated_payoff = [0]*N_f
            mutation_probability = np.zeros(self.UE_num)
            aver_payoff = sum(Satisfaction_vector)/self.UE_num

            for i in range(self.UE_num):
                if Satisfaction_vector[i] < aver_payoff:
                    mutation_probability[i] = 1 - Satisfaction_vector[i]/aver_payoff
                else:
                    mutation_probability[i] = 0
                x = np.random.rand(1)
                # apply mutation rule here
                if x < mutation_probability[i]:
                    old_AP = X_iu_old[i]
                    # find 5 AP candidates for mutation UE
                    SNR_list = self.SINR_matrix[:,i]
                    AP_index = sorted(range(len(SNR_list)), key=lambda i: SNR_list[i], reverse=True)[:N_f]
                    for j in range(N_f):
                        X_iu_old[i] = AP_index[j] + 1 # update X_iu
                        Rho_iu = RA_optimization(self.AP_num, self.UE_num, X_iu_old, self.R_requirement, self.Capacity, opt_mode=RA_mode)
                        Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num
                        sat_list = []
                        for ii in range(self.UE_num):
                            list1 = self.Capacity[:,ii]
                            list2 = Rho_transposed[ii]
                            sat_now = min(sum(list(np.multiply(list1, list2)))/self.R_requirement[ii], 1)
                            sat_list.append(sat_now)
                        estimated_payoff[j] = sum(sat_list)/self.UE_num

                    AP_muted = [AP_index[ii] for ii in range(N_f) if estimated_payoff[ii] == max(estimated_payoff)]

                    if len(AP_muted) > 1:
                        SNR_set = [self.SINR_matrix[k][i] for k in AP_muted]
                        index = SNR_set.index(max(SNR_set))
                        AP_muted_chosen = AP_muted[index]
                        X_iu_old[i] = AP_muted_chosen + 1 # update X_iu here
                    else:
                        X_iu_old[i] = AP_muted[0] + 1 # update X_iu here
                    AP_mutated = X_iu_old[i]

                    if AP_mutated != old_AP:
                        Rho_iu = RA_optimization(self.AP_num, self.UE_num, X_iu_old, self.R_requirement, self.Capacity, opt_mode=RA_mode)
                        Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num

                        Satisfaction_vector = []
                        for ii in range(self.UE_num):
                            list1 = self.Capacity[:,ii]
                            list2 = Rho_transposed[ii]
                            sat_now = min(sum(list(np.multiply(list1, list2)))/self.R_requirement[ii], 1)
                            Satisfaction_vector.append(sat_now)
                        aver_payoff = sum(Satisfaction_vector)/self.UE_num

                        if aver_payoff > payoff_vector[-1]:
                            mode = 0
                            payoff_vector.append(aver_payoff)
                        else:
                            mode += 1
                    else:
                        mode += 1
                else:
                    # no mutation action for UE j
                    payoff_vector.append(payoff_vector[-1])
                if mode > N:
                    break
                count += 1
            if aver_payoff >= 0.99:
                break
        self.X_iu = X_iu_old
        results = [aver_payoff, count, payoff_vector]
        return results

    def JRA(self):
        # iterative JRA solution for each AP, not global solution JRA
        AP_num = self.AP_num
        UE_num = self.UE_num
        Rho_iu = np.zeros((AP_num, UE_num))
        X_iu = np.zeros((AP_num, UE_num), dtype=int)
        for i in range(UE_num):
            X_iu[self.X_iu[i]-1, i] = 1
        # Check if there are any connected UEs
        if np.any(X_iu):
            mask = np.tile(np.eye(AP_num), (1, UE_num))
            # Create constraint matrix A
            X_iu_flat = X_iu.reshape(1, AP_num * UE_num, order='F')
            A = np.tile(X_iu_flat, (AP_num, 1))
            A = A * mask
            B = np.ones((AP_num, 1))
            # Bounds
            LB = np.zeros((AP_num * UE_num, 1))
            UB = np.ones((AP_num * UE_num, 1))
            # Initial guess
            # X0 = (LB + UB) / 2
            X0 = np.random.rand(AP_num * UE_num, 1)
            # Optimization options
            options = {"maxiter": 10000, "ftol": 1e-10}
            # Define constraints
            constraints = [{"type": "ineq", "fun": lambda x: B.flatten() - A @ x}]
            # Perform optimization
            res = minimize(new_obj_function, X0.flatten(),
                        args=(AP_num, UE_num, X_iu, self.Capacity, self.R_requirement),
                        method="SLSQP",
                        bounds=[(lb[0], ub[0]) for lb, ub in zip(LB, UB)],
                        constraints=constraints,
                        options=options)
            X = res.x
            Rho_iu = X.reshape(AP_num, UE_num, order='F')
        # Apply connection mask
        Rho_iu = Rho_iu * X_iu
        return Rho_iu # AP_num * UE_num


def RA_optimization(AP_num, UE_num, X_iu_list, R_required, Capacity, opt_mode):
    # capacity: AP_num * UE_num
    X_iu = np.zeros((AP_num, UE_num), dtype=int)
    for i in range(UE_num):
        X_iu[X_iu_list[i]-1, i] = 1

    Rho_iu = np.zeros((AP_num, UE_num))
    for i in range(AP_num):
        connected_UE = np.where(X_iu[i,:] == 1)[0].tolist()
        if len(connected_UE) != 0:
            if opt_mode == 1:
                A = np.ones((1, len(connected_UE)))
                b = 1
                lb = np.zeros((len(connected_UE),))
                ub = np.ones((len(connected_UE),))
                X0 = (lb+ub)/(len(connected_UE)+1)
                options = {"maxiter": 10000, "ftol": 1e-10}
                Capacity_list = [Capacity[i][j] for j in connected_UE]
                R_list = [R_required[j] for j in connected_UE]
                res = minimize(object_function, X0, args=(Capacity_list, R_list), method="SLSQP", bounds=list(zip(lb,ub)), constraints={"type": "eq", "fun": lambda x: np.dot(A, x) - b}, options=options)
                X = res.x
                Rho_iu[i, connected_UE] = X
            else:
                Rho_iu[i, connected_UE] = np.ones((len(connected_UE))) / len(connected_UE)
    return Rho_iu


def object_function(X, Capacity, R_required):
    cost = 0
    for i in range(len(Capacity)):
        cost += np.log2(min(Capacity[i]*X[i]/R_required[i], 1))
    cost = -cost
    return cost

def new_obj_function(X, AP_num, UE_num, AP_sel, capacity, R_required):
    """
    This objective function considers maximizing the sum of log throughput
    """
    cost = 0
    for UE_ind in range(UE_num):
        coeff_ind = UE_ind * AP_num
        AP_connections = AP_sel[:, UE_ind]
        if np.sum(AP_connections) != 0:
            # Calculate the throughput for this UE
            X_segment = X[coeff_ind:(coeff_ind + AP_num)]
            throughput = np.sum(X_segment * AP_connections * capacity[:, UE_ind])
            normalized_throughput = min(throughput / R_required[UE_ind], 1)
            # Add log throughput to cost
            cost += np.log(normalized_throughput)
        else:
            # No AP connections for this UE, add zero
            X_segment = X[coeff_ind:(coeff_ind + AP_num)]
            cost += np.sum(X_segment) * 0
    # Negate for minimization (since we want to maximize the sum of log throughput)
    return -cost

def mobility_trace(UE_num, x_length, y_length, velocity, total_time):
    ############ generate trace
    slot_num = int(total_time/0.1) + 1
    rw_target = gauss_markov(UE_num, dimensions=(x_length, y_length), velocity_mean=velocity/10, alpha=0.6, variance=0.1)

    user_traces = []
    time_index = 0
    for positions in rw_target:
        if time_index == slot_num:
            break
        else:
            user_traces.append(positions.tolist())  #
            time_index += 1
    UE_traces = []
    for user_id in range(UE_num):
        user_trace = [time_step[user_id] for time_step in user_traces]
        UE_traces.append(user_trace)
    return UE_traces

def plot_trace(UE_traces, user_index):
    # plot the trajectory: trace is a list of [x, y] pairs
    trace_arr = np.array(UE_traces[user_index])
    if trace_arr.size:
        plt.plot(trace_arr[:, 0], trace_arr[:, 1], marker='o', linestyle='-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Trajectory')
        plt.grid(True)
        plt.show()
        plt.savefig('trajectory_user_{}.png'.format(user_index))
    else:
        print("No points to plot.")

def distance(A, B):
    dis = math.sqrt(( float(A[0]) - float(B[0]))**2 + (float(A[1]) - float(B[1]))**2 + (float(A[2]) - float(B[2]))**2)
    return dis

def SINR_calculation(AP_num, UE_num, Signal_power_matrix, interference_matrix):
    # Signal_power_matrix: UE_num * AP_num
    # interference_matrix: AP_num-1 * AP_num-1
    N_0 = 10**(-21) # LiFi Noise power spectral density: Watt/Hz
    B_LiFi = 40*1e6  # unit: Hz
    N_WiFi = -174 # dBm/Hz
    N_WiFi = 10**(N_WiFi/10)/1000  # convert to W/Hz
    B_WiFi = 20*1e6  # unit: Hz
    SINR_matrix = np.zeros((AP_num, UE_num))
    # calculate WiFi SINR
    Signal_power_matrix = np.array(Signal_power_matrix).T # AP_num * UE_num
    SINR_WiFi = Signal_power_matrix[0,:]/(N_WiFi*B_WiFi) # no interference for WiFi
    SINR_matrix[0,:] = SINR_WiFi
    # calculate LiFi SINR
    LiFi_Signal_power_matrix = Signal_power_matrix[1:,:] # (AP_num-1) * UE_num
    for i in range(AP_num - 1):
        for j in range(UE_num):
            signal_power = LiFi_Signal_power_matrix[i, j]
            interference_power = np.multiply(LiFi_Signal_power_matrix[:,j], interference_matrix[i]).sum().item()
            SINR = signal_power / (interference_power + N_0*B_LiFi)
            SINR_matrix[i+1, j] = SINR
    return SINR_matrix.tolist() # AP_num * UE_num

def Signal_power_calculation(X_length, Y_length, Z_height, AP_position, UE_position, mode):
    if mode == 'LiFi':
        # LiFi parameters
        P_mod = 3 # Modulated power
        R_pd = 0.53 # PD responsivity
        Phi = 1.0472 # semiangle: radian
        FOV = 1.3963 # FOV: radian
        n = 1.5 # Reflective index of concentrator
        A = 0.0001 # Detector area: m**2
        m = 1 # Lambertian order
        Ka = 0.8 # optical filter gain
        # LOS
        d_LOS = distance(AP_position, UE_position)
        cos_phi = (Z_height - UE_position[2])/d_LOS
        if abs(math.acos(cos_phi)) <= Phi:
           H_LOS = (m+1)*A*n**2*AP_position[2]**(m+1) / (2*math.pi*(math.sin(FOV))**2*(d_LOS**(m+3))) # correct
        else:
           H_LOS = 0
        # NLOS
        H_NLOS = Capacity_NLOS(AP_position[0], AP_position[1], AP_position[2], UE_position[0], UE_position[1], X_length, Y_length, Z_height) # call sub-function
        signal_power = (R_pd*P_mod*Ka*(H_LOS + H_NLOS))**2
    else:
        # WiFi
        d_LOS = distance(AP_position, UE_position)
        radiation_angle = math.acos(0.5/d_LOS) # radian unit
        P_WiFi_dBm = 20
        P_WiFi = 10**(P_WiFi_dBm/10)/1000 # 20 dBm, convert to watts: 0.1 W
        N_WiFi = -174 # dBm/Hz
        N_WiFi = 10**(N_WiFi/10)/1000  # convert to W/Hz
        f = 2.4*1000000000 # carrier frequency, 2.4 GHz
        # 20 dB loss for concreate wall attenuation
        L_FS = 20*math.log10(d_LOS) + 20*math.log10(f) + 20 - 147.5 # free space loss, unit: dB
        d_BP = 3 # breakpoint distance
        if d_LOS <= d_BP:
            K = 1 # Ricean K-factor
            X = 3 # the shadow fading before breakpoint, unit: dB
            LargeScaleFading = L_FS + X
        else:
            K = 0
            X = 5 # the shadow fading after breakpoint, unit: dB
            LargeScaleFading = L_FS + 35*math.log10(d_LOS/d_BP) + X
        H_WiFi = math.sqrt(K/(K+1))*(math.cos(radiation_angle) + 1j*math.sin(radiation_angle)) + math.sqrt(1/(K+1))*(1/math.sqrt(2)*np.random.rand(1) + 1j/math.sqrt(2)*np.random.rand(1)) # WiFi channel transfer function
        channel =  (abs(H_WiFi))**2 * 10**( -LargeScaleFading / 10 ) # WiFi channel gain
        signal_power = P_WiFi*channel
    return signal_power.item()

def Capacity_NLOS(x_AP, y_AP, z_AP, x_UE, y_UE, X_length, Y_length, Z_height):
    # input x-y-z coordinat of APs to return channel gain H of NLOS
    Phi = (math.pi)/3 # semiangle: radian
    FOV = 80/180*math.pi # FOV: radian
    m = -1/(math.log2(math.cos(Phi))) # Lambertian order
    A = 0.0001 # Detector area: m**2
    n = 1.5 # Reflective index of concentrator
    UE = [x_UE, y_UE, 0] # UE Location
    AP = [x_AP, y_AP, z_AP]
    rho = 0.8 # reflection coefficient of room walls
    step = 0.1   # <--- change from 0.2 to 0.1
    Nx = int(X_length/step)
    Ny = int(Y_length/step)
    Nz = int(Z_height/step) # number of grid in each surface
    X = np.linspace(0, X_length, Nx+1)
    Y = np.linspace(0, Y_length, Ny+1)
    Z = np.linspace(0, Z_height, Nz+1)
    dA = 0.01 # reflective area of wall
    H_NLOS_W1 = [[0]*Nz]*Nx
    H_NLOS_W2 = [[0]*Nz]*Ny
    H_NLOS_W3 = [[0]*Nz]*Nx
    H_NLOS_W4 = [[0]*Nz]*Ny
    for i in range (len(X)-1):
        W1_list = []
        W2_list = []
        W3_list = []
        W4_list = []
        for j in range(len(Z)-1):
            # H11_NLOS of Wall 1 (Left), 1st reflection channel gain between AP1 and UE
            Refl_point_W1 = [0, (Y[i]+Y[i+1])/2, (Z[j]+Z[j+1])/2]
            d1 = distance(AP, Refl_point_W1)
            d2 = distance(UE, Refl_point_W1)
            cos_phi = abs(Refl_point_W1[2] - AP[2])/d1
            cos_alpha = abs(AP[0] - Refl_point_W1[0])/d1
            cos_beta = abs(UE[0] - Refl_point_W1[0])/d2
            cos_psi = abs(UE[2] - Refl_point_W1[2])/d2 # /sai/
            if abs(math.acos(cos_phi)) <= Phi:
                if abs(math.acos(cos_psi)) <= FOV:
                    h = (m+1)*A*rho*dA*cos_phi**m*cos_alpha*cos_beta*cos_psi*n**2/(2*math.pi**2*d1**2*d2**2*(math.sin(FOV))**2)
                else:
                    h = 0
            else:
                h = 0
            W1_list.append(h)

            # H11_NLOS of Wall 2 (Front)
            Refl_point_W2 = [(X[i]+X[i+1])/2, 0, (Z[j]+Z[j+1])/2]
            d1 = distance(AP, Refl_point_W2)
            d2 = distance(UE, Refl_point_W2)
            cos_phi = abs(Refl_point_W2[2]-AP[2])/d1
            cos_alpha = abs(AP[0]-Refl_point_W2[0])/d1
            cos_beta = abs(UE[0]-Refl_point_W2[0])/d2
            cos_psi = abs(UE[2]-Refl_point_W2[2])/d2 # /sai/
            if abs(math.acos(cos_phi)) <= Phi:
                if abs(math.acos(cos_psi)) <= FOV:
                    h = (m+1)*A*rho*dA*cos_phi**m*cos_alpha*cos_beta*cos_psi*n**2/(2*math.pi**2*d1**2*d2**2*(math.sin(FOV))**2)
                else:
                    h = 0
            else:
                h = 0
            W2_list.append(h)

            # H11_NLOS of Wall 3 (Right)
            Refl_point_W3 = [X_length, (Y[i]+Y[i+1])/2, (Z[j]+Z[j+1])/2]
            d1 = distance(AP, Refl_point_W3)
            d2 = distance(UE, Refl_point_W3)
            cos_phi = abs(Refl_point_W3[2]-AP[2])/d1
            cos_alpha = abs(AP[0]-Refl_point_W3[0])/d1
            cos_beta = abs(UE[0]-Refl_point_W3[0])/d2
            cos_psi = abs(UE[2]-Refl_point_W3[2])/d2 # /sai/
            if abs(math.acos(cos_phi)) <= Phi:
                if abs(math.acos(cos_psi)) <= FOV:
                    h = (m+1)*A*rho*dA*cos_phi**m*cos_alpha*cos_beta*cos_psi*n**2/(2*math.pi**2*d1**2*d2**2*(math.sin(FOV))**2)
                else:
                    h = 0
            else:
                h = 0
            W3_list.append(h)

            # H11_NLOS of Wall 4 (Back)
            Refl_point_W4 = [(X[i]+X[i+1])/2, Y_length, (Z[j]+Z[j+1])/2]
            d1 = distance(AP, Refl_point_W4)
            d2 = distance(UE, Refl_point_W4)
            cos_phi = abs(Refl_point_W4[2]-AP[2])/d1
            cos_alpha = abs(AP[0]-Refl_point_W4[0])/d1
            cos_beta = abs(UE[0]-Refl_point_W4[0])/d2
            cos_psi = abs(UE[2]-Refl_point_W4[2])/d2 # /sai/
            if abs(math.acos(cos_phi))<= Phi:
                if abs(math.acos(cos_psi))<= FOV:
                    h = (m+1)*A*rho*dA*cos_phi**m*cos_alpha*cos_beta*cos_psi*n**2/(2*math.pi**2*d1**2*d2**2*(math.sin(FOV))**2)
                else:
                    h = 0
            else:
                h = 0
            W4_list.append(h)
        H_NLOS_W1[i] = W1_list
        H_NLOS_W2[i] = W2_list
        H_NLOS_W3[i] = W3_list
        H_NLOS_W4[i] = W4_list
    H_NLOS = np.sum(H_NLOS_W1) + np.sum(H_NLOS_W2) + np.sum(H_NLOS_W3) + np.sum(H_NLOS_W4)
    return H_NLOS


def collect_dataset(env, UE_positions, time_step, json_file1, json_file2, json_file3):
    # 辅助函数：格式化数值为2位小数并去除逗号
    def format_matrix(matrix):
        if isinstance(matrix, list):
            return [[round(num, 2) for num in row] for row in matrix]
        return matrix

    # 四舍五入并格式化所有数值数据
    sinr_matrix = format_matrix(env.SINR_matrix.T.tolist())
    rho_matrix = format_matrix(env.Rho_iu.T.tolist())
    UE_positions_matrix = format_matrix(UE_positions)
    r_requirement = [round(req, 2) for req in env.R_requirement] if hasattr(env.R_requirement, '__iter__') else round(env.R_requirement, 2)

    ############ Task 1: SINR estimation ############
    os.makedirs(os.path.dirname(json_file1), exist_ok=True)
    if os.path.exists(json_file1):
        try:
            with open(json_file1, 'r', encoding='utf-8') as f:
                dataset_task1 = json.load(f)
        except (json.JSONDecodeError, Exception):
            dataset_task1 = []
    else:
        dataset_task1 = []

    instruction = (
        f"You are a wireless network optimization expert. "
        f"There are {env.AP_num} APs serving {env.UE_num} users in a room of size {env.X_length}m by {env.Y_length}m by {env.Z_height}m. "
        f"There is one WiFi AP and {env.AP_num - 1} LiFi APs, where WiFi AP is in the center of the room with 0.5m above the ground, "
        f"and LiFi APs are placed on the ceiling with positions {env.AP_positions[1:]}. All users are on the ground. "
        f"{env.AP_num} APs are using the frequency reuse in the repeating order of Red, Green, Blue and Yellow to avoid inter-cell interference. "
    )
    input = (
        f"At time step {time_step}, all user's positions are {UE_positions_matrix}. "
        f"Please estimate the signal-to-noise-plus-interference ratio (SINR) in dB for all users, given in a matrix of size {env.UE_num}x{env.AP_num}."
    )
    output = (
        f"SINR matrix is {str(sinr_matrix).replace(',', '')}"  # 去除逗号
    )
    sample_task1 = {
        "instruction": instruction,
        "input": input,
        "output": output
    }
    dataset_task1.append(sample_task1)

    with open(json_file1, 'w', encoding='utf-8') as f:
        json.dump(dataset_task1, f, ensure_ascii=False, indent=2)

    ############ Task 2: access point selection based on GT ############
    os.makedirs(os.path.dirname(json_file2), exist_ok=True)
    if os.path.exists(json_file2):
        try:
            with open(json_file2, 'r', encoding='utf-8') as f:
                dataset_task2 = json.load(f)
        except (json.JSONDecodeError, Exception):
            dataset_task2 = []
    else:
        dataset_task2 = []

    instruction = (
        f"You are a wireless network optimization expert."
        f"There are {env.AP_num} APs serving {env.UE_num} users in a room of size {env.X_length}m by {env.Y_length}m by {env.Z_height}m."
        f"There is one WiFi AP and {env.AP_num - 1} LiFi APs, where WiFi AP is in the center of the room with 0.5m above the ground,"
        f"and LiFi APs are placed on the ceiling with positions {env.AP_positions[1:]}. All users are on the ground."
        f"{env.AP_num} APs are using the frequency reuse in the repeating order of Red, Green, Blue and Yellow to avoid inter-cell interference."
    )
    input = (
        f"At time step {time_step}, all user's SINR matrix in dB is {str(sinr_matrix).replace(',', '')}, and data rate requirement vector for all users is {str(r_requirement).replace(',', '')}."
        f"Please provide the optimal access point selection matrix for all users to obtain highest network capacity."
    )
    output = (
        f"APS result is {str(env.X_iu).replace(',', '')}"  # 去除逗号
    )
    sample_task2 = {
        "instruction": instruction,
        "input": input,
        "output": output
    }
    dataset_task2.append(sample_task2)
    with open(json_file2, 'w', encoding='utf-8') as f:
        json.dump(dataset_task2, f, ensure_ascii=False, indent=2)

    ############ Task 3: resource allocation using JRA algorithm ############
    os.makedirs(os.path.dirname(json_file3), exist_ok=True)
    if os.path.exists(json_file3):
        try:
            with open(json_file3, 'r', encoding='utf-8') as f:
                dataset_task3 = json.load(f)
        except (json.JSONDecodeError, Exception):
            dataset_task3 = []
    else:
        dataset_task3 = []

    instruction = (
        f"You are a wireless network optimization expert."
        f"There are {env.AP_num} APs serving {env.UE_num} users in a room of size {env.X_length}m by {env.Y_length}m by {env.Z_height}m."
        f"There is one WiFi AP and {env.AP_num - 1} LiFi APs, where WiFi AP is in the center of the room with 0.5m above the ground,"
        f"and LiFi APs are placed on the ceiling with positions {env.AP_positions[1:]}. All users are on the ground."
        f"{env.AP_num} APs are using the frequency reuse in the repeating order of Red, Green, Blue and Yellow to avoid inter-cell interference."
    )
    input = (
        f"At time step {time_step}, all user's SINR matrix in dB is {str(sinr_matrix).replace(',', '')}, and data rate requirement vector for all users is {str(r_requirement).replace(',', '')}."
        f"All user's access point selection vector is {str(env.X_iu).replace(',', '')}."
        f"Please provide the optimal resource allocation results in matrix for all APs to obtain highest network capacity."
    )
    output = (
        f"RA result is {str(rho_matrix).replace(',', '')}"  # 去除逗号
    )
    sample_task3 = {
        "instruction": instruction,
        "input": input,
        "output": output
    }
    dataset_task3.append(sample_task3)
    with open(json_file3, 'w', encoding='utf-8') as f:
        json.dump(dataset_task3, f, ensure_ascii=False, indent=2)

    print("Dataset collected for time step ", time_step)


def get_AP_positions(room_mode):
    if room_mode == "Room1-1":
        # 5*5 room, 5 APs (4 LiFi APs) in grid arrangement
        AP_positions = [[2.5, 2.5, 0.5], [1.25, 1.25, 3], [3.75, 1.25, 3], [1.25, 3.75, 3], [3.75, 3.75, 3]]
        interference_matrix = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    elif room_mode == "Room1-2":
        # 5*5 room, 6 APs (5 LiFi APs) in grid arrangement
        AP_positions = [[2.5, 2.5, 0.5], [1.25, 1.25, 3], [3.75, 1.25, 3], [2.5, 2.5, 3], [1.25, 3.75, 3], [3.75, 3.75, 3]]
        interference_matrix = [[0,0,0,0,1], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,0]]
    elif room_mode == "Room2-1":
        # 6*6 room, 5 APs (4 LiFi APs) in grid arrangement
        AP_positions = [[3.0, 3.0, 0.5], [1.5, 1.5, 3], [4.5, 1.5, 3], [1.5, 4.5, 3], [4.5, 4.5, 3]]
        interference_matrix = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    elif room_mode == "Room2-2":
        # 6*6 room, 7 APs (6 LiFi APs) in grid arrangement
        AP_positions = [[3.0, 3.0, 0.5], [1, 1.5, 3], [3, 1.5, 3], [5, 1.5, 3], [1, 4.5, 3], [3, 4.5, 3], [5, 4.5, 3]]
        interference_matrix = [[0,0,0,0,0,1], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [1,0,0,0,0,0]]
    
    
    elif room_mode == "Room3-1":
        # 7*7 room, 6 APs (5 LiFi APs) in grid arrangement
        AP_positions = [[3.5, 3.5, 0.5], [1.75, 1.75, 3], [5.25, 1.75, 3], [3.5, 3.5, 3],[1.75, 5.25, 3], [5.25, 5.25, 3]]
        interference_matrix = [[0,0,0,0,1], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,0]]
    elif room_mode == "Room3-2":
        # 7*7 room, 8 APs (7 LiFi APs) in grid arrangement
        AP_positions = [[3.5, 3.5, 0.5], [7/3, 7/3, 3], [14/3, 7/3, 3], [1.75, 3.5, 3], [3.5, 3.5, 3], [5.25, 3.5, 3], [7/3, 14/3, 3], [14/3, 14/3, 3]]
        interference_matrix = [[0,0,0,0,0,0,1], [0,0,0,0,0,1,0], [0,0,0,0,1,0,0], [0,0,0,0,0,0,0], [0,0,1,0,0,0,0], [0,1,0,0,0,0,0], [1,0,0,0,0,0,0]]
    elif room_mode == "Room3-3":
        # 7*7 room, 5 APs (4 LiFi APs) in grid arrangement
        AP_positions = [[3.5, 3.5, 0.5], [1.75, 1.75, 3], [5.25, 1.75, 3], [1.75, 5.25, 3], [5.25, 5.25, 3]]
        interference_matrix = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    elif room_mode == "Room3-4":
        # 7*7 room, 7 APs (6 LiFi APs) in grid arrangement
        AP_positions = [[3.5, 3.5, 0.5], [7/6, 1.75, 3], [3.5, 1.75, 3], [35/6, 1.75, 3], [7/6, 5.25, 3], [3.5, 5.25, 3], [35/6, 5.25, 3]]
        interference_matrix = [[0,0,0,0,0,1], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0,], [1,0,0,0,0,0]]   

    elif room_mode == "Room4-1":
        # 8*8 room, 7 APs (6 LiFi APs) in grid arrangement
        AP_positions =  [[4.0, 4.0,  0.5], [4/3, 2.0, 3], [4.0, 2.0, 3], [20/3, 2.0, 3], [4/3, 6.0, 3],
                        [4.0, 6.0, 3], [20/3, 6.0, 3]]
        interference_matrix = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0]]  
    elif room_mode == "Room4-2":
        # 8*8 room, 9 APs (8 LiFi APs) in grid arrangement
        AP_positions =  [[4.0, 4.0,  0.5], [4/3, 4/3, 3], [4.0, 4/3, 3], [20/3, 4/3, 3], [8/3, 4.0, 3],
                        [16/3, 4.0, 3], [4/3, 20/3, 3], [4.0, 20/3, 3], [20/3, 20/3, 3]]
        interference_matrix = [[0,0,1,0,0,0,1,0], [0,0,0,0,0,1,0,1], [1,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0], [0,1,0,0,0,1,0,1], [1,0,1,0,0,0,0,0], [0,1,0,0,0,1,0,0]]
    elif room_mode == "Room4-3":
        # 8*8 room, 8 APs (7 LiFi APs) in grid arrangement
        AP_positions =  [[4.0, 4.0,  0.5], [4/3, 2.0, 3], [4.0, 4/3, 3], [20/3, 2.0, 3], [4.0, 4.0, 3],
                        [4/3, 6.0, 3], [4.0, 20/3, 3], [20/3, 6.0, 3]]
        interference_matrix = [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]]  
    elif room_mode == "Room4-4":
        # 8*8 room, 10 APs (9 LiFi APs) in grid arrangement
        AP_positions =  [[4.0, 4.0,  0.5], [2.0, 2.0, 3], [4.0, 2.0, 3], [6.0, 2.0, 3], [2.0, 4.0, 3],
                        [6.0, 4.0, 3], [2.0, 6.0, 3], [4.0, 6.0, 3], [6.0, 6.0, 3], [3.0, 3.0, 3]]
        interference_matrix = [[0,0,0,0,0,1,1,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,1,0,0,0,0,1], [0,0,1,0,0,0,0,0,1],
                                [0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,1,0,0], [1,0,0,0,0,1,0,0,0], [0,1,0,0,0,0,0,0,0],
                                [0,0,1,1,0,0,0,0,0]]   
    elif room_mode == "Room5-1":
        # 9*9 room, 8 APs (7 LiFi APs) in grid arrangement
        AP_positions = [[4.5, 4.5, 0.5], [3, 2.25, 3], [6, 2.25, 3], [2.25, 4.5, 3], [4.5, 4.5, 3],
                        [6.75, 4.5, 3], [3, 6.75, 3], [6, 6.75, 3]]
        interference_matrix = [[0,0,0,0,0,0,1], [0,0,0,0,0,1,0], [0,0,0,0,1,0,0], [0,0,0,0,0,0,0], 
                               [0,0,1,0,0,0,0], [0,1,0,0,0,0,0], [1,0,0,0,0,0,0]]
    elif room_mode == "Room5-2":
        # 9*9 room, 10 APs (9 LiFi APs) in grid arrangement
        AP_positions = [[4.5, 4.5, 0.5], [1.5, 1.5, 3], [4.5, 1.5, 3], [7.5, 1.5, 3], [1.5, 4.5, 3],
                        [7.5, 4.5, 3], [1.5, 7.5, 3], [4.5, 7.5, 3], [7.5, 7.5, 3], [3.0, 3.0, 3]]
        interference_matrix = [[0,0,0,0,0,1,1,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,1,0,0,0,0,1], [0,0,1,0,0,0,0,0,1],
                                [0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,1,0,0], [1,0,0,0,0,1,0,0,0], [0,1,0,0,0,0,0,0,0],
                                [0,0,1,1,0,0,0,0,0]]
    elif room_mode == "Room6-1":
        # 10*10 room, 9 APs (8 LiFi APs) in grid arrangement
        AP_positions = [[5.0, 5.0, 0.5], [5/3, 5/3, 3], [5.0, 5/3, 3], [25/3, 5/3, 3], [10/3, 5.0, 3],
                        [20/3, 5.0, 3], [5/3, 25/3, 3], [5.0, 25/3, 3], [25/3, 25/3, 3]]
        interference_matrix = [[0,0,1,0,0,0,1,0], [0,0,0,0,0,1,0,1], [1,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0], [0,1,0,0,0,1,0,1], [1,0,1,0,0,0,0,0], [0,1,0,0,0,1,0,0]]
    elif room_mode == "Room6-2":
        # 10*10 room, 10 APs (9 LiFi APs) in grid arrangement
        AP_positions = [[5.0, 5.0, 0.5], [2.5, 2.5, 3], [5.0, 2.5, 3], [7.5, 2.5, 3], [2.5, 5.0, 3],
                        [7.5, 5.0, 3], [2.5, 7.5, 3], [5.0, 7.5, 3], [7.5, 7.5, 3], [3.75, 3.75, 3]]
        interference_matrix = [[0,0,0,0,0,1,1,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,1,0,0,0,0,1], [0,0,1,0,0,0,0,0,1],
                                [0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,1,0,0], [1,0,0,0,0,1,0,0,0], [0,1,0,0,0,0,0,0,0],
                                [0,0,1,1,0,0,0,0,0]]
    elif room_mode == "Room7":
        # 5*6 room, 6 APs (5 LiFi APs) in grid arrangement
        AP_positions = [[2.5, 3.0, 0.5], [1.5, 1.25, 3.0], [4.5, 1.25, 3.0],[3, 2.5, 3.0], [1.5, 3.75, 3.0], [4.5, 3.75, 3.0]]
        interference_matrix = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
    elif room_mode == "Room8":
        # 5*8 room, 7 APs (6 LiFi APs) in grid arrangement
        AP_positions = [[2.5, 4.0, 0.5], [4/3, 1.25, 3.0], [4.0, 1.25, 3.0],
                        [20/3, 1.25, 3.0], [4/3, 3.75, 3.0], [4.0, 3.75, 3.0], [20/3, 3.75, 3.0]]
        interference_matrix = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]
    elif room_mode == "Room9":
        # 5*10 room, 8 APs (7 LiFi APs) in grid arrangement
        AP_positions = [[2.5, 5.0, 0.5], [5/3, 1.25, 3.0], [5.0, 5/6, 3.0],
                        [25/3, 1.25, 3.0], [5.0, 2.5, 3.0], [5/3, 3.75, 3.0], [5.0, 25/6, 3.0], [25/3, 3.75, 3.0]]
        interference_matrix = [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]]
    else:
        pass
    return AP_positions, interference_matrix

