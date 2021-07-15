import numpy as np
import torch
import os
import explorer
import utils
import time
import DDPG
import random
import gym
import gym_soccer
# import predictor
from torch.utils.tensorboard import SummaryWriter

def evaluation(env, policy):
	state, done = env.reset(), False
	total_reward = 0
	eps = 0
	while eps < 1000:
		action = policy.select_action(state)
		next_state, reward, done , _= env.step(suit_action(action))
		state = next_state
		total_reward += reward

		if done:
			state, done = env.reset(), False
			eps += 1
	
	return total_reward/1000
		
# ActorのNNから得られるアクションを環境で扱えるように変換
def suit_action(action):
	ret_act = np.zeros(6)
	ret_act[0] = np.argmax(action[0:3])
	ret_act[1:6] = action[3:8]
	return ret_act

# 詳しくはCEの論文と
# onpolicy vs offpolicyの論文：https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/DeepRL16-hausknecht.pdf
# on_policy_mc = Σ^T_i=t(γ^(i-t)*r_i)
# mixing_update: y = beta*on_policy_mc + (1-beta)*q_learning
def add_on_policy_mc(transitions):
	r = 0
	exp_r = 0
	dis = 0.99
	# range(start, stop, step)
	for i in range(len(transitions)-1,-1,-1):
		r = transitions[i]["reward"]+dis*r
		transitions[i]["n_step"] = r
		exp_r = transitions[i]["exp_reward"]+dis*exp_r
		transitions[i]["exp_n_step"] = exp_r

if __name__ == "__main__":
	
	# tensor-board
	# terminalでtensorboard --logdir ./runsを打つとlocalhostのアドレスが出てくるので、webブラウザで結果を見ることができる
	writer = SummaryWriter()

	seed = 0
	save_model = True
	start_timesteps = 1000
	# ネットワークのトレーニングする際にサンプリングするバッチサイズ
	batch_size = 256

	file_name = "DDPG_" + "HFO_" + str(seed)
	print("---------------------------------------")
	print(f"Policy: DDPG, Env: HFO, Seed: {seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	torch.manual_seed(seed)
	np.random.seed(seed)
	

	# アクションの最大と最小
	max_a = [1,1,1,100,180,180,100,180]
	min_a = [-1,-1,-1,0,-180,-180,0,-180]
	state_dim = 59
	action_dim = len(max_a)

	# DDPGのインスタンス化
	policy = DDPG.DDPG(state_dim, action_dim, max_a, min_a)
	# Curious Explorerのインスタンス化
	# Curious Explorationの論文：https://arxiv.org/abs/2105.00499
	explore = explorer.explorer(state_dim, action_dim, max_a, min_a)


	# Experience Replayで使用するリプレイバッファのインスタンス化
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	# もらえる報酬がゴール時のみのタスク（スパース報酬）
	env = gym.make('Soccer-v0')
	state, done = env.reset(), False
	episode_reward = 0
	exp_episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	transitions = []
	high_eval = 0
	timestep = 0
	evaluation_num = 0
	# epsilon-greedyをアニーリングする,rewardがもらえて、dec>0.1のときdec-=0.001
	dec = 0
	# exp_reward重み付けのハイパーパラメータ
	ro = 0.003
	while True:
		# epsilon-greedyを使用するが、Curious Explorationなため常に探索を行う eps_rnd < dec, start_timestepsまでは探索のみを行う
		# eps_rnd = random.random()
		if timestep < start_timesteps:
			action = policy.select_action(state)
		else:
			action, dec = explore.select_action(state)

		# action[0]の0~4までが離散アクション,パラメータ：DASH[1],[2],TURN[3],KICK[4],[5]
		next_state, reward, done ,info= env.step(suit_action(action))
		# アニーリング
		#if reward > 0 and dec > 0.1:
		#	print('decreased it')
		#	dec -= 0.001

		# CEでの予測結果
		predicted_state = explore.predict(state, action)

		# doneを数値化
		done_bool = float(done)
		# exp_reward = ||(St+1, Rt+1) - P(St, at)||^2
		exp_reward = np.linalg.norm(np.concatenate((next_state,np.array([reward])))-predicted_state)
		exp_reward = ro * exp_reward
		# CEの報酬をCLIPする
		#if exp_reward > 0.5:
		#	exp_reward = 0.5
		transitions.append({"state" : state,
							"action" : action,
							"next_state" : next_state,
							"reward" : reward,
							"exp_reward" : exp_reward,
							"done" : done_bool
							})

		state = next_state
		episode_reward += reward
		exp_episode_reward +=  exp_reward

		# timestepごとのintrinsic rewards
		writer.add_scalar("exp_reward/timestep",exp_reward,timestep)
		writer.add_scalar("dec/timestep",dec, timestep)
		timestep += 1
		episode_timesteps+=1

		# Episode終了
		if done:
			# モンテカルロアップデートで使うものを以下でtransitionに付け加える
			add_on_policy_mc(transitions)
			# transitionをすべてreplay bufferへいれる
			for i in transitions:
				replay_buffer.add(i["state"], i["action"], i["next_state"],
									i["reward"], i["exp_reward"], i["n_step"],
									i["exp_n_step"], i["done"])
			# tensorboard用辞書の初期化
			debug_dict = {"predictor_loss":0,"current_q":0,"mixed_q":0,"critic_loss":0}
			# timestepが十分な探索を超えたらトレーニングを始める
			if timestep >= start_timesteps:
				for i in range(int(episode_timesteps/10)):
					policy.train(replay_buffer, batch_size)
					critic_q_and_loss, predictor_loss = explore.train(replay_buffer,batch_size)
					debug_dict["predictor_loss"] += predictor_loss
					debug_dict["current_q"] += critic_q_and_loss[0]
					debug_dict["mixed_q"] += critic_q_and_loss[1]
					debug_dict["critic_loss"] += critic_q_and_loss[2]

			writer.add_scalar("reward/episode", episode_reward, episode_num)
			writer.add_scalar("predictor_loss/episode", debug_dict["predictor_loss"], episode_num)
			writer.add_scalar("exp_reward/episode",exp_episode_reward,episode_num)
			writer.add_scalar("current_q/episode",debug_dict["current_q"], episode_num)
			writer.add_scalar("mixed_q/episode",debug_dict["mixed_q"], episode_num)
			writer.add_scalar("critic_loss/episode",debug_dict["critic_loss"], episode_num)

			# エピソード終了のリセット
			state, done = env.reset(), False
			episode_reward = 0
			exp_episode_reward =0
			transitions = []
			episode_timesteps = 0
			episode_num += 1 

			# エピソードが10000ごとに評価をする
			if (episode_num+1) % 10000 == 0 :
				evaluation_num += 1
				current_eval = evaluation(env, explore.ddpg)
				print('evaluation : ', current_eval)
				writer.add_scalar("current_eval/test_number", current_eval, evaluation_num)
				#if current_eval > high_eval:
				policy.save('./models/policy_model_{}'.format(episode_num+1))
				explore.ddpg.save('./models/exploer_model_{}'.format(episode_num+1))
				explore.predictor.save('./models/predictor_{}'.format(episode_num+1))
				replay_buffer.save('./memory',episode_num+1)
				high_eval = current_eval
				print('saved in ',episode_num)
				#env.close()
				#time.sleep(3)
				#break
	writer.flush()