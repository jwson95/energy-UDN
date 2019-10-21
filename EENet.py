import tensorflow as tf

#######################################################################################################
# EENet_RNN은 L time slots동안의 channel을 차례대로 받아 L slot이 끝났을 때 L slot동안 소모했던 power consumption을 계산하여
# 이를 minimize하는 방향으로 training한다.
#######################################################################################################

class EENet_RNN(object):
    def __init__(self, sess, name, regularizer_rate, regularizer_on, batch_size, base_num, user_num, power_small, power_macro, layer_width, sequence_length, P_on, P_turnon, noise_variance, onoff_crit, amplifier_efficiency):
        self.sess = sess
        self.name = name
        self.regularizer_rate = regularizer_rate
        self.regularizer_on = regularizer_on
        self.batch_size = batch_size
        self.base = base_num # small AP 와 macro cell 합친 수
        self.user = user_num
        self.power_small = power_small
        self.power_macro = power_macro
        self.width = layer_width
        self.sequence_length = sequence_length # total time slot length
        self.P_on = P_on # 켜져 있을 때 고정으로 나가는 파워
        self.P_turnon = P_turnon # 킬 때 들어가는 파워
        self.noise_variance = noise_variance
        self.onoff_crit = onoff_crit # power가 몇 이상이어야 on으로 볼 것인가
        self.amplifier_efficiency = amplifier_efficiency
        self._build_net()
        
    def _build_net(self):
        with tf.variable_scope(self.name):
            self.learning_rate = tf.placeholder(tf.float32, shape = ())
            self.isTraining = tf.placeholder(tf.bool, name = 'isTraining')
            self.minimum_rate = tf.placeholder(tf.float32, shape = ())
            
            self.input_channel = tf.placeholder(tf.float32, [None, self.sequence_length, self.user*self.base])
            self.input_channel_rescale = self.input_channel * 5e13
            self.input_channel_log = (tf.log(self.input_channel_rescale)/tf.log(tf.constant(10,dtype = tf.float32))-1)/3*1 - 0.2
            self.input_channel_standard = tf.contrib.layers.instance_norm(self.input_channel)
            
            self.input_rate = tf.tile([self.minimum_rate], [tf.cast(tf.size(self.input_channel)/(self.sequence_length*self.user*self.base),tf.int32) * self.sequence_length])
            self.input_rate = tf.reshape(self.input_rate, [tf.cast(tf.size(self.input_channel)/(self.sequence_length*self.user*self.base),tf.int32), self.sequence_length, 1])
            
#            self.input = tf.concat([self.input_channel_log, self.input_rate], axis = 2)
            self.distance = tf.placeholder(tf.float32, [None, self.sequence_length, self.user*self.base])
            self.input = self.distance / 1000
            ##################################################################################################
            
            W0 = tf.get_variable("W0", shape=[self.user * self.base, self.width], initializer = tf.contrib.layers.xavier_initializer())
            b0 = tf.get_variable("b0", shape= [self.width], initializer = tf.constant_initializer(0.001))
            L0 = tf.tensordot(self.input, W0, axes = [[2], [0]]) + b0
            self.inputRNN = tf.nn.relu(tf.contrib.layers.layer_norm(L0, begin_norm_axis = 1))
            
            cell_LSTM = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units = self.width)
            cell_LSTM2 = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units = self.width)
            cell_LSTM3 = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units = self.width)
#            self.multi_cells = tf.nn.rnn_cell.MultiRNNCell([cell_LSTM, cell_LSTM2, cell_LSTM3], state_is_tuple=True)
            self.multi_cells = tf.nn.rnn_cell.MultiRNNCell([cell_LSTM], state_is_tuple=True)
            self.first_state = self.multi_cells.zero_state(batch_size = tf.cast(tf.size(self.input_channel)/(self.sequence_length*self.user*self.base),tf.int32), dtype = tf.float32)
            self.output_RNN, self._states = tf.nn.dynamic_rnn(self.multi_cells, self.inputRNN, initial_state = self.first_state, dtype = tf.float32)
#            self.output_RNN = self.inputRNN
            
            W2 = tf.get_variable("W2", shape = [self.width, self.width], initializer = tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable("b2", shape= [self.width], initializer = tf.constant_initializer(value = 0))
            S2 = tf.tensordot(self.output_RNN, W2, axes = [[2], [0]]) + b2
#            L2 = tf.nn.relu(S2)
            L2 = tf.nn.relu(tf.contrib.layers.layer_norm(S2, begin_norm_axis = 1))
            
            W3 = tf.get_variable("W3", shape = [self.width, self.width], initializer = tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable("b3", shape= [self.width], initializer = tf.constant_initializer(value = 0))
            S3 = tf.tensordot(L2, W3, axes = [[2], [0]]) + b3
#            L3 = tf.nn.relu(S3)
            L3 = tf.nn.relu(tf.contrib.layers.layer_norm(S3, begin_norm_axis = 1))

            W4 = tf.get_variable("W4", shape = [self.width, self.width], initializer = tf.contrib.layers.xavier_initializer())
            b4 = tf.get_variable("b4", shape= [self.width], initializer = tf.constant_initializer(value = 0))
            S4 = tf.tensordot(L3, W4, axes = [[2], [0]]) + b4
#            L4 = tf.nn.relu(S4)
            L4 = tf.nn.relu(tf.contrib.layers.layer_norm(S4, begin_norm_axis = 1))
            
            W5 = tf.get_variable("W5", shape = [self.width, self.width], initializer = tf.contrib.layers.xavier_initializer())
            b5 = tf.get_variable("b5", shape= [self.width], initializer = tf.constant_initializer(value = 0))
            S5 = tf.tensordot(L4, W5, axes = [[2], [0]]) + b5
#            L5 = tf.nn.relu(S5)
            L5 = tf.nn.relu(tf.contrib.layers.layer_norm(S5, begin_norm_axis = 1))
            
            ##################
            
            self.Wsigmoidon = tf.get_variable("Wsigmoidon", shape=[self.width, self.base-1], initializer = tf.contrib.layers.xavier_initializer())
            bsigmoidon = tf.get_variable("bsigmoidon", shape= [self.base-1], initializer = tf.constant_initializer(value = 0))
#            self.Ssigmoidon = tf.nn.sigmoid(tf.tensordot(L5, self.Wsigmoidon, axes = [[2], [0]]) + bsigmoidon)
            self.Ssigmoidon = tf.nn.sigmoid((tf.tensordot(L5, self.Wsigmoidon, axes = [[2], [0]]) + bsigmoidon)*4)

            self.Wmacroon = tf.get_variable("Wmacroon", shape=[self.width, 1], initializer = tf.contrib.layers.xavier_initializer())
            bmacroon = tf.get_variable("bmacroon", shape= [1], initializer = tf.constant_initializer(value = 0))
#            self.Smacroon = tf.nn.sigmoid(tf.tensordot(L5, self.Wmacroon, axes = [[2], [0]]) + bmacroon)
            self.Smacroon = tf.nn.sigmoid((tf.tensordot(L5, self.Wmacroon, axes = [[2], [0]]) + bmacroon)*4)
#            self.Smacroon = tf.constant(1.0, shape = [5000, 5, 1])
            
            self.Wsigmoidp = tf.get_variable("Wsigmoidp", shape=[self.width, self.base-1], initializer = tf.contrib.layers.xavier_initializer())
            bsigmoidp = tf.get_variable("bsigmoidp", shape= [self.base-1], initializer = tf.constant_initializer(value = 0))
            self.Ssigmoidp0to1 = tf.nn.sigmoid(tf.tensordot(L5, self.Wsigmoidp, axes = [[2], [0]]) + bsigmoidp)
            Ssigmoidp = self.Ssigmoidp0to1 * self.power_small
            
            self.Wmacrop = tf.get_variable("Wmacrop", shape=[self.width, 1], initializer = tf.contrib.layers.xavier_initializer())
            bmacrop = tf.get_variable("bmacrop", shape= [1], initializer = tf.constant_initializer(value = 0))
            self.Smacrop0to1 = tf.nn.sigmoid(tf.tensordot(L5, self.Wmacrop, axes = [[2], [0]]) + bmacrop)
            Smacrop = self.Smacrop0to1 * self.power_macro
            
            
            ############################################################################################### softmax 별개의 neural network로
#            self.Wsoftmax = tf.get_variable("Wsoftmax", shape=[self.width, self.base * self.user], initializer = tf.contrib.layers.xavier_initializer())
#            bsoftmax = tf.get_variable("bsoftmax", shape= [self.base * self.user], initializer = tf.constant_initializer(value = 0))
#            self.Lsoftmax = tf.nn.softmax(tf.reshape(tf.tensordot(L5,self.Wsoftmax, axes = [[2], [0]])+bsoftmax, [-1, self.sequence_length, self.user, self.base]), axis = 2)
##            self.Lsoftmax = tf.nn.softmax(tf.reshape(self.input_channel_rescale, [-1,self.sequence_length,self.user,self.base]), axis = 2)

#            Seffective = self.Son * Sp
#            Leffective_diag = tf.matrix_diag(Seffective)
#            Leffective = tf.reshape(Leffective_diag,[-1, self.sequence_length, self.base, self.base])
#            self.Llast = tf.matmul(self.Lsoftmax, Leffective) # dimension : (batch) x (sequence_length) x (user) x (base)
            
#            Sreal = self.Son_bin * Sp
#            Lreal_diag = tf.matrix_diag(Sreal)
#            Lreal = tf.reshape(Lreal_diag,[-1, self.sequence_length, self.base, self.base])
#            self.Llast_real = tf.matmul(self.Lsoftmax, Lreal)
            ###############################################################################################

            self.Wpowersig = tf.get_variable("Wpowersig", shape=[self.width, self.base * self.user], initializer = tf.contrib.layers.xavier_initializer())
            bpowersig = tf.get_variable("bpowersig", shape= [self.base * self.user], initializer = tf.constant_initializer(value = 0))
            self.Lpowersig = tf.nn.sigmoid(tf.tensordot(L5,self.Wpowersig, axes = [[2],[0]]) + bpowersig) * self.power_small
            self.Lpowersig_reshape = tf.reshape(self.Lpowersig,  [-1, self.sequence_length, self.user, self.base]) # dimension : (batch) x (sequence_length) x (user) x (base)
            
            input_reshape = tf.reshape(self.input_channel, [-1, self.sequence_length, self.user, self.base])
            
            self.Son = tf.concat([self.Ssigmoidon, self.Smacroon], axis = 2)
            self.Son_bin = (tf.sign(self.Son-self.onoff_crit)+1)/2

            self.Llast = tf.matmul(self.Lpowersig_reshape, tf.reshape(tf.matrix_diag(self.Son),[-1, self.sequence_length, self.base, self.base]))
            
            desired_signal = tf.reduce_sum(tf.multiply(input_reshape, self.Llast),3) # dimension : (batch) x (seqeunce_length) x (user)
            self.Linterference = tf.reduce_sum(self.Llast, axis = 2, keep_dims= True) - self.Llast # dimension : (batch) x (sequence_length) x (user) x (base) / meaning : 각 유저에게 오는 interference의 합
            interference = tf.reduce_sum(tf.multiply(input_reshape, self.Linterference),3) # dimension : (batch) x (seqeunce_length) x (user)
            
            self.rate_constraint = desired_signal - (tf.pow(2.0, self.minimum_rate)-1)*interference - self.noise_variance**2*(tf.pow(2.0, self.minimum_rate)-1)
            
            self.Llast_real = tf.matmul(self.Lpowersig_reshape, tf.reshape(tf.matrix_diag(self.Son_bin),[-1, self.sequence_length, self.base, self.base]))

            desired_signal_real = tf.reduce_sum(tf.multiply(input_reshape, self.Llast_real),3) # dimension : (batch) x (seqeunce_length) x (user)
            Linterference_real = tf.reduce_sum(self.Llast_real, axis = 2, keep_dims= True) - self.Llast_real # dimension : (batch) x (sequence_length) x (user) x (base) / meaning : 각 유저에게 오늘 interference의 합
            interference_real = tf.reduce_sum(tf.multiply(input_reshape, Linterference_real),3) # dimension : (batch) x (seqeunce_length) x (user)
            
            self.achievable_rate = tf.log(1+desired_signal_real/(self.noise_variance**2 + interference_real))/tf.log(2.0)
            
            ############################################################################################# Loss function

            self.P_transmission = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(self.Llast_real,3),2),1),0)/self.amplifier_efficiency
            self.C_transmission = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(self.Llast,3),2),1),0)/self.amplifier_efficiency
            
            self.C_turned_on = (tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(self.Ssigmoidon, 2),1),0) + tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(self.Smacroon*10, 2),1),0))* self.P_on
            self.P_turned_on = (tf.reduce_mean(tf.reduce_sum(tf.reduce_sum((tf.sign(self.Ssigmoidon-self.onoff_crit)+1)/2, 2),1),0) + tf.reduce_mean(tf.reduce_sum(tf.reduce_sum((tf.sign(self.Smacroon-self.onoff_crit)+1)/2*10, 2),1),0)) * self.P_on
            self.N_turned_on = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum((tf.sign(self.Ssigmoidon-self.onoff_crit)+1)/2, 2),1),0) + tf.reduce_mean(tf.reduce_sum(tf.reduce_sum((tf.sign(self.Smacroon-self.onoff_crit)+1)/2, 2),1),0)
#            self.L_turned_on = tf.concat([(tf.sign(self.Ssigmoidon-self.onoff_crit)+1)/2, (tf.sign(self.Smacroon-self.onoff_crit)+1)/2],)
            
            self.C_turning_on = (tf.reduce_mean(tf.reduce_sum(tf.reduce_sum((-self.Ssigmoidon[:,0:self.sequence_length-1,:]+1)*self.Ssigmoidon[:,1:self.sequence_length,:],2),1),0) + tf.reduce_mean(tf.reduce_sum(tf.reduce_sum((-self.Smacroon[:,0:self.sequence_length-1,:]+1)*self.Smacroon[:,1:self.sequence_length,:]*10,2),1),0))*self.P_turnon
            self.P_turning_on = (tf.reduce_mean(tf.reduce_sum(tf.reduce_sum((-tf.sign(self.Ssigmoidon[:,0:self.sequence_length-1,:]-self.onoff_crit)+1)/2*(tf.sign(self.Ssigmoidon[:,1:self.sequence_length,:]-self.onoff_crit)+1)/2,2),1),0) + tf.reduce_mean(tf.reduce_sum(tf.reduce_sum((-tf.sign(self.Smacroon[:,0:self.sequence_length-1,:]-self.onoff_crit)+1)/2*(tf.sign(self.Smacroon[:,1:self.sequence_length,:]-self.onoff_crit)+1)/2*10,2),1),0))*self.P_turnon
            
            self.C_turning_off = (tf.reduce_mean(tf.reduce_sum(tf.reduce_sum((-self.Ssigmoidon[:,1:self.sequence_length,:]+1)*self.Ssigmoidon[:,0:self.sequence_length-1,:],2),1),0) + tf.reduce_mean(tf.reduce_sum(tf.reduce_sum((-self.Smacroon[:,1:self.sequence_length,:]+1)*self.Smacroon[:,0:self.sequence_length-1,:]*10,2),1),0))*self.P_turnon
            self.P_turning_off = (tf.reduce_mean(tf.reduce_sum(tf.reduce_sum((-tf.sign(self.Ssigmoidon[:,1:self.sequence_length,:]-self.onoff_crit)+1)/2*(tf.sign(self.Ssigmoidon[:,0:self.sequence_length-1,:]-self.onoff_crit)+1)/2,2),1),0) + tf.reduce_mean(tf.reduce_sum(tf.reduce_sum((-tf.sign(self.Smacroon[:,1:self.sequence_length,:]-self.onoff_crit)+1)/2*(tf.sign(self.Smacroon[:,0:self.sequence_length-1,:]-self.onoff_crit)+1)/2*10,2),1),0))*self.P_turnon

            self.rate_now = tf.log(1+desired_signal/(self.noise_variance**2 + interference))/tf.log(2.0)
            self.C_rate = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.pow(tf.nn.relu(self.minimum_rate - self.rate_now),2),2),1),0)*self.regularizer_rate
            
            self.C_maxp = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.nn.relu(tf.reduce_sum(self.Lpowersig_reshape, 3) - self.power_small),2),1),0) * 1e2
            
            self.cost = self.C_transmission + self.C_turned_on + self.C_turning_on + self.C_turning_off + self.C_rate + self.C_maxp
            
            ###############################################################################################
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
#            self.optimizer = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = 0.9).minimize(self.cost)
            
            self.parameters = tf.trainable_variables()
            
    def train(self, input_channel, learning_rate, minimum_rate, distance):
        return self.sess.run([self.cost, self.optimizer, self.achievable_rate, self.Llast, self.Son, self.P_transmission, self.P_turned_on, self.P_turning_on, self.P_turning_off, self.C_rate, self.N_turned_on, self.Son_bin], feed_dict={self.input_channel : input_channel, self.isTraining: True, self.learning_rate: learning_rate, self.minimum_rate : minimum_rate, self.distance : distance})

    def test(self, input_channel, minimum_rate, distance):
        return self.sess.run([self.cost, self.achievable_rate, self.Llast, self.Son, self.P_transmission, self.P_turned_on, self.P_turning_on, self.P_turning_off, self.C_rate, self.N_turned_on, self.Son_bin, self.inputRNN, self.input_channel, self.input_channel_log, self.input_channel_standard, self.C_maxp, self.Lpowersig_reshape], feed_dict={self.input_channel : input_channel, self.isTraining: False, self.minimum_rate : minimum_rate, self.distance : distance})

    def load(self, saver, save_file):
        saver.restore(self.sess, save_file)