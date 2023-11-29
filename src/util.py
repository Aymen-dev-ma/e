import numpy as np
import pickle
from .game_environment import Game
from .tfmodel import ActiveInferenceModel

np_precision = np.float32

def make_batch_dsprites_random(game, index, size, repeats):
    o0 = np.zeros((size, 64, 64, 1), dtype=np_precision)
    o1 = np.zeros((size, 64, 64, 1), dtype=np_precision)
    S0_real = np.zeros((size, 6), dtype=np_precision)
    S1_real = np.zeros((size, 6), dtype=np_precision)
    pi_one_hot = np.zeros((size,4), dtype=np_precision)
    for i in range(size):
        game.randomize_environment(index)
        o0[i] = game.current_frame(index)
        S0_real[i] = game.current_s[index,1:]
        S0_real[i,5] = game.last_r[index]
        Ppi = np.random.rand(4).astype(np_precision)
        Ppi /= np.sum(Ppi)
        pi0 = np.random.choice(4, p=Ppi)
        game.pi_to_action(pi0, index, repeats=repeats)
        pi_one_hot[i, pi0] = 1.0
        o1[i] = game.current_frame(index)
        S1_real[i] = game.current_s[index,1:]
        S1_real[i,5] = game.last_r[index]
    return o0, o1, pi_one_hot, S0_real, S1_real

def make_batch_dsprites_random_reward_transitions(game, index, size, deepness=1, repeats=1):
    '''
    Make a batch of random datapoints which are designed to test whether the
    agent understands the concept of reward changes..
    '''
    o0 = np.zeros((size, 64, 64, 1), dtype=np_precision)
    o1 = np.zeros((size, 64, 64, 1), dtype=np_precision)
    pi0 = np.zeros((size),dtype=np.int32) # just 'up'
    pi_one_hot = np.zeros((size,4), dtype=np_precision)
    for i in range(size):
        game.randomize_environment(index)
        game.current_s[index,5] = 31 # Object located right at the edge of crossing.
        o0[i] = game.current_frame(index)
        for t in range(deepness):
            game.pi_to_action(pi0[i], index, repeats=repeats)
        pi_one_hot[i,pi0[i]] = 1.0
        o1[i] = game.current_frame(index)
    return o0, o1, pi_one_hot

def softmax_multi_with_log(x, single_values=4, eps=1e-20, temperature=10.0):
    """Compute softmax values for each sets of scores in x."""
    x = x.reshape(-1, single_values)
    x = x - np.max(x,1).reshape(-1,1) # Normalization
    e_x = np.exp(x/temperature)
    SM = e_x / e_x.sum(axis=1).reshape(-1,1)
    logSM = x - np.log(e_x.sum(axis=1).reshape(-1,1) + eps) # to avoid infs
    return SM, logSM

def make_batch_dsprites_active_inference(
        games: Game,
        model: ActiveInferenceModel,
        deepness: int = 10,
        samples: int = 5,
        calc_mean: bool = False,
        repeats: int = 1,
    ):
    o0 = games.current_frame_all()
    # TODO: Find out why is this Magic Number 4? should this be pi_dim?
    #o0_repeated = o0.repeat(4,0) # The 0th dimension
    o0_repeated = o0.repeat(model.pi_dim, 0)


    pi_one_hot = np.eye(model.pi_dim)
    pi_repeated = np.tile(pi_one_hot,(games.games_no, 1))

    sum_G, sum_terms, po2 = model.calculate_G_repeated(o0_repeated, pi_repeated, steps=deepness, samples=samples, calc_mean=calc_mean)
    terms1 = -sum_terms[0]
    terms12 = -sum_terms[0]+sum_terms[1]
    # Shape now is (games_no,4)
    #Ppi, log_Ppi = softmax_multi_with_log(-terms1.numpy(), 4) # For agent driven just by reward
    #Ppi, log_Ppi = softmax_multi_with_log(-terms12.numpy(), 4) # For agent driven by terms 1 and 2
    Ppi, log_Ppi = softmax_multi_with_log(-sum_G.numpy(), 4) # Full active inference agent

    pi_choices = np.array([np.random.choice(4,p=Ppi[i]) for i in range(games.games_no)])

    # One hot version..
    pi0 = np.zeros((games.games_no,4), dtype=np_precision)
    pi0[np.arange(games.games_no), pi_choices] = 1.0

    # Apply the actions!
    for i in range(games.games_no): games.pi_to_action(pi_choices[i], i, repeats=repeats)
    o1 = games.current_frame_all()

    return o0, o1, pi0, log_Ppi

def compare_reward(o1, po1):
    ''' Using MSE. '''
    logpo1 = np.square(o1 - po1).mean()
    #logpo1 = np.square(o1[:,0:3,0:64,:] - po1[:,0:3,0:64,:]).mean(axis=(0,1,2,3))
    return logpo1

def generate_perlin_noise(n_iter, timescale, baseline=20, scaling=5):
    class PerlinNoiseGenerator:
        def __init__(self):
            self.baseline  = baseline
            self.scaling   = scaling
            self.n_iter    = n_iter
            self.timescale = timescale

        def fade(self, t):
            """Fade function as defined by Ken Perlin."""
            return t * t * t * (t * (t * 6 - 15) + 10)

        def lerp(self, a, b, x):
            """Linear interpolation."""
            return a + x * (b - a)

        def grad(self, hash_value, x):
            """Gradient function."""
            h = hash_value & 15
            grad = 1 + (h & 7)  # Gradient value is one of 1, 2, ..., 8
            return (grad * x)  # Compute the dot product

        def generate(self):
            # Create a permutation array
            p = np.arange(256, dtype=int)
            np.random.shuffle(p)
            p = np.stack([p, p]).flatten()  # Duplicate to avoid overflow

            def perlin(x):
                """Generate Perlin noise for input x."""
                X = int(x) & 255
                x -= int(x)
                u = self.fade(x)

                a = p[X]
                b = p[X + 1]

                return self.lerp(self.grad(a, x), self.grad(b, x - 1), u)

            # Example of using the Perlin noise generator to produce temperature variations
            time_steps = np.linspace(0, self.timescale, self.n_iter)
            outdoor_temperatures = [20 + self.scaling*perlin(t) for t in time_steps]
            return time_steps, outdoor_temperatures

    time_steps, outdoor_temps = PerlinNoiseGenerator().generate()
    return time_steps, outdoor_temps

class NoiseClass:
    def __init__(self, scale=100, x_offset=None, mean=20, dev=10):
        """
        Initialize the NoiseClass with a given scale and starting offset.
        :param scale: Scale of the noise.
        :param x_offset: Initial offset in the noise field.
        """
        self.scale = scale
        self.x_offset = x_offset
        if self.x_offset is None:
            self.x_offset = np.random.uniform(-10000, 10000)  # Random starting point
        self.mean = mean
        self.dev  = dev

    def next(self):
        """
        Generate the next Simplex noise value and increment the offset.
        :return: Next noise value.
        """
        noise_value = snoise2(self.x_offset / self.scale, 0)
        self.x_offset += 1
        return self.mean + self.dev*noise_value