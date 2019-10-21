import numpy as np

#%% parameters
learning_rate = 1e-4
batch_size = 5000
test_size = 100
layer_width = 512
base_num = 9 # small cell만 count
user_num = 6
#minimum_rate = 0.1
noise_variance = 2e-3
onoff_crit = 0.2
training_epochs = 10001
check_period = 50
regularizer_rate = 5e3/ (noise_variance ** 2)
regularizer_on = 1e2
P_on = 2
P_turnon = 1
power_small = 2
power_macro = 10
sequence_length = 5

# path loss를 계산하기 위한 parameter들은 cell-free massive 논문 참조
frequency = 1900 # MHz 단위
h_base = 15 # meter 단위
h_user = 1.65 # meter 단위
d1 = 50 # meter 단위
d0 = 10 # meter 단위

#%%
def gen_data():
    speed = 0.001
    base_loc_x = np.array([[0.8997],[0.3083],[0.2356],[0.2001],[0.4923],[0.8949],[0.6571],[0.5100],[0.1]])
    base_loc_y = np.array([[0.7061],[0.7454],[0.4210],[0.1066],[0.9235],[0.1342],[0.8120],[0.2314],[0.9]])
    macro_loc_x = 0.5
    macro_loc_y = 0.5
    large_scale_coeff_batch = np.ones((batch_size,sequence_length,(base_num+1)*user_num))
    distance_batch = np.zeros((batch_size,sequence_length, (base_num+1)*user_num))
    step_batch = 0
    while step_batch < batch_size:
        large_scale_coeff_time = np.ones((sequence_length,(base_num+1)*user_num))
        large_scale_coeff_vec = np.ones((1,(base_num+1)*user_num))
        user_loc_x = np.random.rand(user_num,1)
        user_loc_y = np.random.rand(user_num,1)
        speed_x = speed * np.random.rand(user_num,1)
        speed_y = speed * np.random.rand(user_num,1)
        for step_time in range(sequence_length):
            distance = (((user_loc_x - base_loc_x.transpose())**2)+((user_loc_y - base_loc_y.transpose())**2))**(1/2)
            distance = np.append(distance, (((user_loc_x - macro_loc_x)**2)+((user_loc_y - macro_loc_y)**2))**(1/2),1)
            distance_vec = np.reshape(distance, (1,user_num*(base_num+1))) * 1e3
            L = 46.3 + 33.9*np.log10(frequency)-13.82*np.log10(h_base)-(1.1*np.log10(frequency)-0.7)*h_user+(1.56*np.log10(frequency)-0.8)
                
            PL_dB = np.zeros((1,(base_num+1)*user_num))
            for step_distance in range(user_num*(base_num+1)):
                if distance_vec[0,step_distance] > d1:
                    PL_dB[0,step_distance] = -L -35*np.log10(distance_vec[0,step_distance])
                elif distance_vec[0,step_distance] > d0:
                    PL_dB[0,step_distance] = -L -15*np.log10(d1) -20*np.log10(distance_vec[0,step_distance])
                else:
                    PL_dB[0,step_distance] = -L -15*np.log10(d1) -20*np.log10(d0)
            large_scale_coeff_vec = np.power(10,(PL_dB / 20))
            large_scale_coeff_time[step_time,:] = large_scale_coeff_vec
            distance_batch[step_batch,step_time,:] = distance_vec
                
            user_loc_x = user_loc_x + speed_x
            user_loc_y = user_loc_y + speed_y
        large_scale_coeff_batch[step_batch,:,:] = large_scale_coeff_time
        step_batch = step_batch + 1
    input_channel = large_scale_coeff_batch
    return input_channel, distance_batch

data_length = 1
file_channel = np.memmap('C:/Users/jwson/OneDrive - SNU/code/python/cell-less/save/data_channel_test2', dtype = 'float32', mode='w+', shape=(batch_size, sequence_length, (base_num+1)*user_num))
file_distance = np.memmap('C:/Users/jwson/OneDrive - SNU/code/python/cell-less/save/data_distance_test2', dtype = 'float32', mode='w+', shape=(batch_size, sequence_length, (base_num+1)*user_num))
#data_length = 1000
#file_channel = np.memmap('C:/Users/jwson/OneDrive - SNU/code/python/cell-less/save/data_channel_train', dtype = 'float32', mode='w+', shape=(batch_size * data_length, sequence_length, (base_num+1)*user_num))
#file_distance = np.memmap('C:/Users/jwson/OneDrive - SNU/code/python/cell-less/save/data_distance_train', dtype = 'float32', mode='w+', shape=(batch_size * data_length, sequence_length, (base_num+1)*user_num))


for step in range(data_length):
    channel_step,distance_step = gen_data()
    file_channel[(step*batch_size):((step+1)*batch_size),:,:] = channel_step
    file_distance[(step*batch_size):((step+1)*batch_size),:,:] = distance_step
    print('%d' % step)