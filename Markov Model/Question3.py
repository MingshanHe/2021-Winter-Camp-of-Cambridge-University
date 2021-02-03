# /*
#  * @Author: MingshanHe
#  * @Email: hemingshan@robotics.github.com
#  * @Date: 2021-02-03 15:15:59
#  * @Last Modified by:   MingshanHe
#  * @Last Modified time: 2021-02-03 15:15:59
#  * @Description: This code for Question3
#  */
import numpy as np

# 对应状态集合Q
states = ('Sunny', 'Rainny')
# 对应观测集合V
observations = ('Shorts', 'Jacket', 'Umbrella')
# 初始状态概率向量π
start_probability = {'Sunny': 0.6, 'Rainny': 0.4}
# 状态转移矩阵A
transition_probability = {
    'Sunny' : {'Sunny': 0.7, 'Rainny': 0.3},
    'Rainny': {'Sunny': 0.4, 'Rainny': 0.6},
}
# 观测概率矩阵B
emission_probability = {
    'Sunny' : {'Shorts': 0.60, 'Jacket': 0.30, 'Umbrella': 0.10},
    'Rainny': {'Shorts': 0.05, 'Jacket': 0.55, 'Umbrella': 0.40},
}

# 随机生成观测序列和状态序列    
def simulate(T):

    def draw_from(probs):
        """
        1.np.random.multinomial:
        按照多项式分布，生成数据
        >>> np.random.multinomial(20, [1/6.]*6, size=2)
                array([[3, 4, 3, 3, 4, 3],
                       [2, 4, 3, 4, 0, 7]])
         For the first run, we threw 3 times 1, 4 times 2, etc.  
         For the second, we threw 2 times 1, 4 times 2, etc.
        2.np.where:
        >>> x = np.arange(9.).reshape(3, 3)
        >>> np.where( x > 5 )
        (array([2, 2, 2]), array([0, 1, 2]))
        """
        return np.where(np.random.multinomial(1,probs) == 1)[0][0]

    observations = np.zeros(T, dtype=int)
    states = np.zeros(T, dtype=int)
    states[0] = draw_from(pi)
    observations[0] = draw_from(B[states[0],:])
    for t in range(1, T):
        states[t] = draw_from(A[states[t-1],:])
        observations[t] = draw_from(B[states[t],:])
    return observations, states

def generate_index_map(lables):
    id2label = {}
    label2id = {}
    i = 0
    for l in lables:
        id2label[i] = l
        label2id[l] = i
        i += 1
    return id2label, label2id
 
states_id2label, states_label2id              =  generate_index_map(states)
observations_id2label, observations_label2id  =  generate_index_map(observations)
print(states_id2label, states_label2id)
print(observations_id2label, observations_label2id)

def convert_map_to_vector(map_, label2id):
    """将概率向量从dict转换成一维array"""
    v = np.zeros(len(map_), dtype=float)
    for e in map_:
        v[label2id[e]] = map_[e]
    return v

 
def convert_map_to_matrix(map_, label2id1, label2id2):
    """将概率转移矩阵从dict转换成矩阵"""
    m = np.zeros((len(label2id1), len(label2id2)), dtype=float)
    for line in map_:
        for col in map_[line]:
            m[label2id1[line]][label2id2[col]] = map_[line][col]
    return m

A = convert_map_to_matrix(transition_probability, states_label2id, states_label2id)
print(A)
B = convert_map_to_matrix(emission_probability, states_label2id, observations_label2id)
print(B)
observations_index = [observations_label2id[o] for o in observations]
pi = convert_map_to_vector(start_probability, states_label2id)
print(pi)

# 生成模拟数据
observations_data, states_data = simulate(10)
print(observations_data)
print(states_data)
# 相应的label
print("天气的状态: ", [states_id2label[index] for index in states_data])
print("穿着的观测: ", [observations_id2label[index] for index in observations_data])

#Problem.1: Calculating Probility
def forward(obs_seq):
    """前向算法"""
    N = A.shape[0]
    T = len(obs_seq)
    
    # F保存前向概率矩阵
    F = np.zeros((N,T))
    F[:,0] = pi * B[:, obs_seq[0]]

/*
 * @Author: MingshanHe
 * @Email: hemingshan@robotics.github.com
 * @Date: 2021-02-03 15:15:55
 * @Last Modified by: MingshanHe
 * @Last Modified time: 2021-02-03 15:16:15
 * @Description: Description
 */
    for t in range(1, T):
        for n in range(N):
            F[n,t] = np.dot(F[:,t-1], (A[:,n])) * B[n, obs_seq[t]]

    return F

def backward(obs_seq):
    """后向算法"""
    N = A.shape[0]
    T = len(obs_seq)
    # X保存后向概率矩阵
    X = np.zeros((N,T))
    X[:,-1:] = 1

    for t in reversed(range(T-1)):
        for n in range(N):
            X[n,t] = np.sum(X[:,t+1] * A[n,:] * B[:, obs_seq[t+1]])

    return X

cloth = [0,1,1]
day = forward(cloth)
print(day)