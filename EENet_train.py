import tensorflow as tf
import numpy as np
import math
import time
from scipy.optimize import linprog 
from EENet import EENet_RNN

tf.reset_default_graph()

#%% parameters
learning_rate = 1e-4
batch_size = 5000
test_size = 5000
layer_width = 512
base_num = 9
user_num = 4
#minimum_rate = 0.1
noise_variance = 0.000002 * math.sqrt(1) ################################
onoff_crit = 0.01
training_epochs = 300001
check_period = 1000
regularizer_rate = 1/ (noise_variance ** 2) / 1000000000
regularizer_on = 1e2
P_on = 2
P_turnon = 1
power_small = 2
power_macro = 10
sequence_length = 5
amplifier_efficiency = 0.4
shadow_variance = 0

#%%
def optimize_linprog(user_num, base_num, onoff_bin, block_index, large_scale_coeff_vec, minimum_rate_test):
    a = onoff_bin[block_index,:]
    large_scale_coeff = np.transpose(np.reshape(large_scale_coeff_vec[block_index,:], (user_num, base_num + 1))) # dimension : (base) x (user)
    data_rate_min = minimum_rate_test * np.ones([user_num])
    SINR_min = np.zeros([user_num])
    SINR_min[:] = np.exp2(data_rate_min[:])-1
    P_max_array = power_small*np.append(np.ones([base_num]), np.array([10]))

    beta = np.zeros([np.count_nonzero(a)*user_num, user_num])
    for k in range(user_num):
        for l in range(user_num):
            if not l==k:
                beta[l*np.count_nonzero(a):(l+1)*np.count_nonzero(a), k] = -SINR_min[k]*large_scale_coeff[np.nonzero(a),k]
            else:
                beta[k*np.count_nonzero(a):(k+1)*np.count_nonzero(a), k] = large_scale_coeff[np.nonzero(a),k]
                        
    lambda1 =np.zeros([np.count_nonzero(a)*user_num, np.count_nonzero(a)])

    for m in range(np.count_nonzero(a)):
        for k in range(user_num):
            lambda1[k*np.count_nonzero(a)+m, m] = 1

    c = np.ones([np.count_nonzero(a)*user_num]) # output res.x의 dimension과 같음 (계수임) / res.x은 승년이형 코드에서는 Gamma로 나타남
    
    A = np.append(-beta,lambda1,axis=1)
    b = np.append(-(noise_variance**2)*SINR_min, P_max_array[np.nonzero(a)])
    res = linprog(c, A_ub = np.transpose(A), b_ub = np.transpose(b))
    
    if res.status == 0:
        P_trans_linprog = res.fun / amplifier_efficiency
        solved = 1
    else:
        P_trans_linprog = 0
        solved = 0
        
    return P_trans_linprog, res.status, res.x, solved

#%%
def gen_data(isTraining, epoch):
    epoch2 = epoch % 1000
    if isTraining == False:
        large_scale_coeff_batch = np.copy(openfile_test)
        input_channel = large_scale_coeff_batch
        distance = np.copy(distance_test)
    else:
        large_scale_coeff_batch = openfile_train[epoch2*batch_size:(epoch2+1)*batch_size,:,:]
        input_channel = large_scale_coeff_batch
        distance = distance_train[epoch2*batch_size:(epoch2+1)*batch_size,:,:]
    minimum_rate = (np.random.randint(1)+20)/100
    return input_channel, minimum_rate, distance
 
#%%
def calculate_rate(large_scale_coeff_vec, power_coeff_on, onoff_bin, block_index):
    base_on = np.reshape(np.nonzero(onoff_bin[block_index,:]), (np.count_nonzero(onoff_bin[block_index,:])))
    power_coeff = np.zeros([user_num, base_num+1])
    power_coeff[:,base_on] = np.reshape(power_coeff_on, (user_num, np.count_nonzero(onoff_bin[block_index,:])))
    large_scale_coeff = np.reshape(large_scale_coeff_vec[block_index,:], (user_num, base_num + 1))
    
    achievable_rate = np.zeros([user_num])
    for step_user in range(user_num):
        power_interference = np.copy(power_coeff)
        power_interference[step_user,:] = np.zeros([1,base_num+1])
        interference = np.sum(power_interference, axis = 0) * large_scale_coeff[step_user,:] # interference
        desired_signal = power_coeff[step_user,:] * large_scale_coeff[step_user,:] # desired_power
        achievable_rate[step_user] = np.log2(1+np.sum(desired_signal)/(noise_variance**2 + np.sum(interference)))
        
    return achievable_rate

#%%
sess = tf.Session()
Model_RNN = EENet_RNN(sess, "EENet", regularizer_rate, regularizer_on, batch_size, base_num+1, user_num, power_small, power_macro, layer_width, sequence_length, P_on, P_turnon, noise_variance, onoff_crit, amplifier_efficiency)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
save_file = 'C:/Users/jwson/OneDrive - SNU/code/python/cell-less/save/modelsave'
#Model_RNN.load(saver,save_file)

#%% training
t0 = time.time()
openfile_train = np.memmap('C:/Users/jwson/OneDrive - SNU/code/python/cell-less/save/data_channel_train_4', dtype = 'float32', mode='r+', shape=(100 * batch_size, sequence_length, (base_num+1)*user_num))
openfile_test = np.memmap('C:/Users/jwson/OneDrive - SNU/code/python/cell-less/save/data_channel_test_4', dtype = 'float32', mode='r+', shape=(batch_size, sequence_length, (base_num+1)*user_num))
distance_train = np.memmap('C:/Users/jwson/OneDrive - SNU/code/python/cell-less/save/data_distance_train_4', dtype = 'float32', mode='r+', shape=(100 * batch_size, sequence_length, (base_num+1)*user_num))
distance_test = np.memmap('C:/Users/jwson/OneDrive - SNU/code/python/cell-less/save/data_distance_test_4', dtype = 'float32', mode='r+', shape=(batch_size, sequence_length, (base_num+1)*user_num))

record_size = int((training_epochs-1)/check_period)
cost_train_record = np.zeros(record_size)
cost_test_record = np.zeros(record_size)
power_test_record = np.zeros(record_size)
power_so_far = 1000
number_so_far = 50
for epoch in range(training_epochs):
    input_channel,minimum_rate, distance = gen_data(True, epoch)
    cost, _, achievable_rate_val, Llast_train, sigmoid_train, P_transmission_train, P_fixed_train, P_turningon_train, P_turningoff_train, C_rate_train, N_on_train, Son_bin_train = Model_RNN.train(input_channel = input_channel, learning_rate = learning_rate, minimum_rate = minimum_rate, distance = distance)
    if epoch % check_period == 0:
        input_channel, minimum_rate_test, distance = gen_data(False, epoch)
        cost_test, achievable_rate_val_test, Llast_test, sigmoid_test, P_transmission_test, P_fixed_test, P_turningon_test, P_turningoff_test, C_rate_test, N_on_test, Son_bin_test, inputRNN, inputlin, inputlog, inputstd, C_maxp, a = Model_RNN.test(input_channel = input_channel, minimum_rate = minimum_rate_test, distance = distance)
        total_power_val_train = P_fixed_train + P_turningon_train + P_turningoff_train + P_transmission_train
        total_power_val_nn = P_fixed_test + P_turningon_test + P_turningoff_test + P_transmission_test
        
        solved_matrix = np.zeros([test_size, sequence_length])
        solved_vector = np.zeros([test_size])
        solved = 0
        P_transmission_linprog = 0
        
        ##########################################
        achievable_rate_linprog = np.zeros([test_size, sequence_length, user_num])
        
        distance_error = np.empty((0,(base_num+1) * user_num), float) # rate 만족 못시키는 경우의 데이터 확인용
        input_channel_error = np.empty((0,(base_num+1) * user_num), float)
        P_alloc = []
        for step_test in range(test_size):
            large_scale_coeff = input_channel[step_test,:,:]
            onoff_bin = Son_bin_test[step_test,:,:]
            for step_sequence in range(sequence_length):
                P_transmission_linprog_step, _, P_alloc_vec, solved_step = optimize_linprog(user_num, base_num, onoff_bin, step_sequence, large_scale_coeff, minimum_rate_test)
                P_alloc.append(P_alloc_vec)
                solved_matrix[step_test, step_sequence] = solved_step
                solved = solved + solved_step
                P_transmission_linprog = P_transmission_linprog + P_transmission_linprog_step
                if solved_step == 0:
                    distance_error = np.concatenate((distance_error, np.reshape(distance[step_test, step_sequence,:],(1,(base_num+1)*user_num))), axis = 0)
                    input_channel_error = np.concatenate((input_channel_error, np.reshape(input_channel[step_test, step_sequence,:],(1,(base_num+1)*user_num))), axis = 0)
            if all(solved_matrix[step_test,:] == np.ones(sequence_length)):
                solved_vector[step_test] = 1
        
        if solved == 0:
            P_transmission_linprog = 0
        else:
            P_transmission_linprog = P_transmission_linprog/solved*sequence_length
        
        total_power_val_linprog = P_fixed_test + P_turningon_test + P_turningoff_test + P_transmission_linprog
        
        print("EPOCH : ", '%05d' % epoch, 'TOTAL POWER(TRAIN) = %.5f' % total_power_val_train, 'N_on = %.4f' % N_on_test, 'RATE = %.2f' % minimum_rate,  "TIME : %.2f" % (time.time() - t0))
        print('COST(TRAIN) = %.5f' % cost, 'COST_RATE(TRAIN) = %.5f' % C_rate_train, 'COST_MAXP = %.5f' % C_maxp, 'noise = %.8f' % noise_variance)
        print('COST(TEST) = %.5f' % cost_test, 'COST_RATE(TEST) = %.5f' % C_rate_test, 'solved = %3d' % solved, 'solved_vector = %3d' % np.sum(solved_vector), 'POWER(NN) = %.3f' % total_power_val_nn, 'POWER(LP) = %.3f' % total_power_val_linprog)
        print("----------------------------------------------------------------------------------------------------------------------------")
        t0 = time.time()
        
        record_step = int(epoch/check_period - epoch%check_period)
        cost_train_record[record_step] = cost
        cost_test_record[record_step] = cost_test
        power_test_record[record_step] = total_power_val_linprog
        if np.sum(solved_vector) == 5000 and total_power_val_linprog < power_so_far:
            power_so_far = total_power_val_linprog
            number_so_far = N_on_test
            saver.save(sess, save_file)
sess.close()