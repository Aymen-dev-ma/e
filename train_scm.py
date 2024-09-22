import os
import time
import numpy as np
import argparse
import tensorflow as tf
from sys import argv
from distutils.dir_util import copy_tree

# Suppress TensorFlow logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import custom libraries
from src.game_environment import Game
import src.util as u
import src.tfloss as loss
from src.tfutils import *
from graphs.reconstructions_plot import reconstructions_plot
from graphs.generate_traversals import generate_traversals
from graphs.stats_plot import stats_plot

# Import Advanced Causal Inference Model
from src.tfmodel import AdvancedCausalInferenceModel  # Ensure the updated class is used here

parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('-r', '--resume', action='store_true', help='Resume training from existing weights.')
parser.add_argument('-b', '--batch', type=int, default=50, help='Select batch size.')
args = parser.parse_args()

var_a = 1.0
var_b = 25.0
var_c = 5.0
var_d = 1.5
s_dim = 10
pi_dim = 4
beta_s = 1.0
beta_o = 1.0
gamma = 0.0
gamma_rate = 0.01
gamma_max = 0.8
gamma_delay = 30
deepness = 1
samples = 1
repeats = 5
l_rate_top = 1e-04
l_rate_mid = 1e-04
l_rate_down = 0.001
ROUNDS = 1000
TEST_SIZE = 1000
epochs = 60  # Adjust as needed

signature = 'causal_backdoor_model_'  
signature += f"{gamma_rate}_{gamma_delay}_{var_a}_{args.batch}_{s_dim}_{repeats}"
folder = 'figs_' + signature
folder_chp = folder + '/checkpoints'

os.makedirs(folder, exist_ok=True)
os.makedirs(folder_chp, exist_ok=True)

games = Game(args.batch)
game_test = Game(1)
model = AdvancedCausalInferenceModel(
    s_dim=s_dim,
    pi_dim=pi_dim,
    gamma=gamma,
    beta_s=beta_s,
    beta_o=beta_o,
    colour_channels=1,
    resolution=64
)

stats_start = {
    'F': [], 'F_top': [], 'F_mid': [], 'F_down': [], 'mse_o': [], 'TC': [], 'kl_div_s': [],
    'omega': [], 'learning_rate': [], 'current_lr': [], 'mse_r': [], 'omega_std': [],
    'kl_div_pi': [], 'var_beta_s': [], 'var_gamma': [], 'var_beta_o': [], 'score': []
}

if args.resume:
    stats, optimizers = model.load_all(folder_chp)
    start_epoch = len(stats['F']) + 1
else:
    stats = stats_start
    start_epoch = 1
    optimizers = {}

if not optimizers:
    optimizers['top'] = tf.keras.optimizers.Adam(learning_rate=l_rate_top)
    optimizers['mid'] = tf.keras.optimizers.Adam(learning_rate=l_rate_mid)
    optimizers['down'] = tf.keras.optimizers.Adam(learning_rate=l_rate_down)

start_time = time.time()
for epoch in range(start_epoch, epochs + 1):
    if epoch > gamma_delay and model.model_down.gamma < gamma_max:
        model.model_down.gamma.assign(model.model_down.gamma + gamma_rate)

    for i in range(ROUNDS):
        games.randomize_environment_all()
        o0 = games.new_image_all()
        o0_tf = tf.convert_to_tensor(o0, dtype=tf.float32)

        # Choose actions using backdoor adjustment
        pi0 = []
        for obs in o0_tf:
            action = model.choose_action_with_backdoor(tf.expand_dims(obs, axis=0))
            pi0.append(action)
        pi0 = np.array(pi0)

        pi_indices = np.argmax(pi0, axis=1)
        games.perform_actions(pi_indices)
        o1 = games.new_image_all()
        o1_tf = tf.convert_to_tensor(o1, dtype=tf.float32)

        model.update_on_transition(o0_tf, pi0, o1_tf)

        log_Ppi = np.log(pi0 + 1e-15)
        qs0, _, _ = model.model_down.encoder_with_sample(o0_tf)
        D_KL_pi = loss.train_model_top(model_top=model.model_top, s=qs0, log_Ppi=log_Ppi, optimizer=optimizers['top'])
        current_omega = loss.compute_omega(D_KL_pi.numpy(), a=var_a, b=var_b, c=var_c, d=var_d).reshape(-1, 1)

        qs1_mean, qs1_logvar = model.model_down.encoder(o1_tf)
        ps1_mean, ps1_logvar = loss.train_model_mid(
            model_mid=model.model_mid,
            s0=qs0,
            qs1_mean=qs1_mean,
            qs1_logvar=qs1_logvar,
            Ppi_sampled=pi0,
            omega=current_omega,
            optimizer=optimizers['mid']
        )

        loss.train_model_down(
            model_down=model.model_down,
            o1=o1_tf,
            ps1_mean=ps1_mean,
            ps1_logvar=ps1_logvar,
            omega=current_omega,
            optimizer=optimizers['down']
        )

    if epoch % 2 == 0:
        model.save_all(folder_chp, stats)
    if epoch % 25 == 0:
        copy_tree(folder_chp, f"{folder_chp}_epoch_{epoch}")

    # Add evaluation code here similar to the original script

    print(f"{epoch}, F: {stats['F'][-1]:.2f}, mse_o: {stats['mse_o'][-1]:.3f}, kl_s: {stats['kl_div_s'][-1]:.2f}, "
          f"omega: {stats['omega'][-1]:.2f}, TC: {stats['TC'][-1]:.2f}, duration: {round(time.time() - start_time, 2)}s")
    start_time = time.time()
