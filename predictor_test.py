import numpy as np
import torch
import torch.nn.functional as F
import predictor
import utils
import matplotlib.pyplot as plt

'''
外部報酬が上がったときのpredictorとその後下がったときのpredictorの誤差の違いを見たい。
そのエピソードで溜まっていたメモリのMSEを出力する
'''


def eval_mse(model, replay_buffer):
    state, action, next_state, reward, ex_reward, n_step, ex_n_step, not_done = replay_buffer.sample(100000)
    loss = F.mse_loss(model.net(state, action), torch.cat((next_state, reward), 1))
    return loss

def eval_intrinsic_reward(model, state, action, next_state, reward, int_reward):
    for i in range(500000):
        int_reward[i] = np.linalg.norm(np.concatenate((next_state[i], reward[i])) - model.predict(state[i], action[i])).clip(max=30)
        if i % 10000 == 0:
            print("finished {}".format(i))

def main():
    # アクションの最大と最小
    max_a = [1,1,1,100,180,180,100,180]
    min_a = [-1,-1,-1,0,-180,-180,0,-180]
    state_dim = 59
    action_dim = len(max_a)

    # modelとmemoryのフォルダーパス
    model_folder = './models/'
    memory_folder = './memory'

    # Load memory
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    memory_episode_num = 10000
    state = np.load(memory_folder + '/state_{}.npy'.format(memory_episode_num))
    action = np.load(memory_folder + '/action_{}.npy'.format(memory_episode_num))
    next_state = np.load(memory_folder + '/next_state_{}.npy'.format(memory_episode_num))
    reward = np.load(memory_folder + '/reward_{}.npy'.format(memory_episode_num))
    replay_buffer.load(folder=memory_folder,episode_num=memory_episode_num)
    print(replay_buffer)

    # Load predictor
    predictor_episode_num = 200000
    file_name = 'predictor_{}'.format(predictor_episode_num)
    predictor_model = predictor.Predictor(state_dim, action_dim)
    predictor_model.load(filename=model_folder+file_name)
    print(predictor_model)

    intrinsic_rewards = np.zeros((500000,1))
    eval_intrinsic_reward(predictor_model, state, action, next_state, reward, intrinsic_rewards)
    plt.plot(np.linspace(0,500000,500000), intrinsic_rewards, label='intrinsic_reward')
    plt.legend()
    plt.show()

    loss = eval_mse(predictor_model, replay_buffer)

if __name__ == '__main__':
    main()