import time
import numpy as np
from src.util import np_precision, generate_perlin_noise, NoiseClass

REWARD_IDX = 4

class Game:
    def __init__(self, number_of_games=1, pi_dim=5):
        current_time = time.time()
        #dataset = np.load('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',allow_pickle=True,encoding='latin1')
        #self.imgs = dataset['imgs'].reshape(-1,64,64,1)
        self.game_length = 1000
        #latents_values = dataset['latents_values'] # Normalized version of classes..
        #latents_classes = dataset['latents_classes']

        # State dimensions
        #self.s_sizes = metadata['latents_sizes'] # [1 3 6 40 32 32]
        #self.s_bases = np.concatenate((self.s_sizes[::-1].cumprod()[::-1][1:], np.array([1,], dtype=np_precision))) # [737280 245760  40960 1024 32 1]
        #self.s_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY', 'reward']
        self.s_names = ['outdoor_temp', 'indoor_temp', 'agent_action', 'agent_power', 'reward']
        self.s_sizes = np.ones_like(self.s_names, dtype=float)
        self.s_dim   = self.s_sizes.size
        self.s_key   = dict([(key, idx) for (idx, key) in enumerate(self.s_names)])
        self.s_bases = None # hope this isn't used!!

        # Obesrvation Dimensions
        self.o_names = ["indoor_temp", "agent_power"]
        self.o_sizes = np.ones_like(self.o_names, dtype=float)
        self.o_dim   = self.o_sizes.size
        self.o_key   = dict([(key, idx) for (idx, key) in enumerate(self.o_names)])

        self.pi_dim  = pi_dim

        self.games_no = number_of_games
        self.weather = np.zeros((self.games_no, self.game_length))
        self.current_s = np.zeros((self.games_no, self.s_dim), dtype=np_precision)
        self.last_r = np.zeros(self.games_no, dtype=np_precision)
        self.new_image_all()
        #print('Dataset loaded. Time:', time.time() - current_time, 'datapoints:', len(self.imgs), self.s_dim, self.s_bases)
        print("Game Loaded. Time: ", time.time() - current_time)

    def sample_s(self): # Reward is zero after this!
        s = np.zeros(self.s_dim, dtype=np_precision)
        for s_i, s_size in enumerate(self.s_sizes):
            s[s_i] = np.random.randint(s_size)
        return s

    def sample_s_all(self): # Reward is zero after this!
        s = np.zeros((self.games_no,self.s_dim), dtype=np_precision)
        for i in range(self.games_no):
            s[i] = self.sample_s()
        return s

    #def s_to_index(self, s):
    #    return np.dot(s, self.s_bases).astype(int)

    def s_to_o(self, index):
        indoor_temp  = self.current_s[index, self.s_key["indoor_temp"]]
        agent_action = self.current_s[index, self.s_key["agent_action"]]

        # TODO: calculate reward
        reward = 0.0
        obs_to_return = np.array([ indoor_temp, agent_action, reward ])

        return obs_to_return

        #image_to_return = self.imgs[self.s_to_index(self.current_s[index,:-1])].astype(np.float32)

        # Adding the reward encoded to the image..
        #if 0.0 <= self.last_r[index] <= 1.0:
        #    image_to_return[0:3,0:32] = self.last_r[index]
        #elif -1.0 <= self.last_r[index] < 0.0:
        #    image_to_return[0:3,32:64] = -self.last_r[index]
        #else:
        #    exit('Error: Reward: '+str(self.last_r[index]))
        #return image_to_return

    #def reward_to_rgb(self, reward):
    #    return np.array([ min(1.0, -reward+1), min(1.0, reward+1), 1.0 - abs(reward)], dtype=np_precision)

    def current_frame(self, index):
        return self.s_to_o(index)

    def current_frame_all(self):
        o = np.zeros((self.games_no, self.o_dim), dtype=np_precision)
        for i in range(self.games_no):
            o[i] = self.s_to_o(i)
        return o

    def randomize_environment(self,index):
        # perform randomization
        self.current_s[index] = self.sample_s()
        self.weather[index] = NoiseClass()

        # woah make it a bit less random
        self.current_s[index, self.s_key["outdoor_temp"]] = self.weather[index].next()
        self.current_s[index, REWARD_IDX] = -10 + np.random.rand()*20
        self.last_r[index] = -1.0 + np.random.rand()*2.0

    def randomize_environment_all(self):
        for index in range(self.games_no):
            self.randomize_environment(index)

    def new_image(self, index):
        reward = self.current_s[index, self.s_key["reward"]] # pass reward to the new latent..!
        self.current_s[index] = self.sample_s()
        self.current_s[index, self.s_key["reward"]] = reward

    def new_image_all(self):
        reward = self.current_s[:, self.s_key["reward"]] # pass reward to the new latent..!
        self.current_s = self.sample_s_all()
        self.current_s[:, self.s_key["reward"]] = reward

    def get_reward(self, index):
        return self.current_s[index, self.s_key["reward"]]

    # NOTE: Randomness takes values from zero to one.
    def find_move(self, index, randomness):
        Ppi = np.ones(self.pi_dim) // np.sum(np.ones(self.pi_dim))
        return Ppi

    def find_move_all(self, randomness):
        return np.array([self.find_move(i, randomness) for i in range(self.games_no)], dtype=np_precision)

    # NOTE: Randomness takes values from zero to one.
    def auto_play(self, index, randomness=0.4):
        Ppi = self.find_move(index, randomness)
        pi = np.random.choice(self.pi_dim, p=Ppi)
        self.pi_to_action(pi, index)
        return pi, Ppi

    def tick(self, index):
        self.last_r[index] *= 0.95

    def tick_all(self):
        self.last_r *= 0.95

    def update_temp(self, index):
        self.tick(index)
        weather = self.weather[index]
        outdoor_temp = weather.next()
        indoor_temp  = self.current_s[index, self.s_key["indoor_temp"]]
        agent_power  = self.current_s[index, self.s_key["agent_power"]]

        # Update temperature
        outdoor_temp = weather.next()
        indoor_temp  = indoor_temp \
            + (outdoor_temp - indoor_temp)*0.02 \
            + agent_power*0.2 \
            + np.random.normal(0, 0.1)

        # TODO: check this is correct?
        self.last_r[index] = 1 - ((indoor_temp-20.0)/10.0)**2
        self.current_s[index, self.s_key["reward"]] += self.last_r[index]

        self.current_s[index, self.s_key["outdoor_temp"]] = outdoor_temp
        self.current_s[index, self.s_key["indoor_temp"]]  = indoor_temp

    def pi_to_action(self, pi, index, repeats=1, p_change=0.02):
        strengths = np.linspace(-1, 1, self.pi_dim)
        for i in range(repeats):
            self.current_s[self.s_key["agent_action"]] = pi
            self.current_s[self.s_key["agent_power"]] = strengths[pi]
            self.update_temp(index)

        if np.random.uniform(0.0, 1.0) < p_change:
            self.new_image(index)
            return True
        return False
