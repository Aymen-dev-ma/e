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

# Import the new AdvancedBackdoorAdjustmentModel
from src.tfmodel import AdvancedBackdoorAdjustmentModel  # Ensure this model is implemented

parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('-r', '--resume', action='store_true', help='If this is used, the script tries to load existing weights and resume training.')
parser.add_argument('-b', '--batch', type=int, default=50, help='Select batch size.')
args = parser.parse_args()

# Set initial hyperparameters and constants
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
epochs = 60  # Number of training epochs

signature = 'backdoor_adjustment_model_'
signature += f"{gamma_rate}_{gamma_delay}_{var_a}_{args.batch}_{s_dim}_{repeats}"
folder = 'figs_' + signature
folder_chp = folder + '/checkpoints'

# Create necessary directories
os.makedirs(folder, exist_ok=True)
os.makedirs(folder_chp, exist_ok=True)

# Initialize the environment and model
games = Game(args.batch)
game_test = Game(1)
model = AdvancedBackdoorAdjustmentModel(
    s_dim=s_dim,
    pi_dim=pi_dim,
    gamma=gamma,
    beta_s=beta_s,
    beta_o=beta_o,
    colour_channels=1,
    resolution=64
)

# Initialize stats to track during training
stats_start = {
    'F': [], 'F_top': [], 'F_mid': [], 'F_down': [], 'mse_o': [], 'TC': [], 'kl_div_s': [],
    'kl_div_s_anal': [], 'omega': [], 'learning_rate': [], 'current_lr': [], 'mse_r': [],
    'omega_std': [], 'kl_div_pi': [], 'kl_div_pi_min': [], 'kl_div_pi_max': [],
    'kl_div_pi_med': [], 'kl_div_pi_std': [], 'kl_div_pi_anal': [], 'deep_mse_o': [],
    'var_beta_o': [], 'var_beta_s': [], 'var_gamma': [], 'var_a': [], 'var_b': [],
    'var_c': [], 'var_d': [], 'kl_div_s_naive': [], 'kl_div_s_naive_anal': [], 'score': [],
    'train_scores_m': [], 'train_scores_std': [], 'train_scores_sem': [], 'train_scores_min': [],
    'train_scores_max': []
}

# Load weights if resuming from a checkpoint
if args.resume:
    stats, optimizers = model.load_all(folder_chp)
    for k in stats_start.keys():
        if k not in stats:
            stats[k] = []
        while len(stats[k]) < len(stats['F']):
            stats[k].append(0.0)
    start_epoch = len(stats['F']) + 1
else:
    stats = stats_start
    start_epoch = 1
    optimizers = {}

# Initialize optimizers if not resumed
if not optimizers:
    optimizers['top'] = tf.keras.optimizers.Adam(learning_rate=l_rate_top)
    optimizers['mid'] = tf.keras.optimizers.Adam(learning_rate=l_rate_mid)
    optimizers['down'] = tf.keras.optimizers.Adam(learning_rate=l_rate_down)

# Start training loop
start_time = time.time()
for epoch in range(start_epoch, epochs + 1):
    # Increment gamma after a delay, but cap it
    if epoch > gamma_delay and model.model_down.gamma < gamma_max:
        model.model_down.gamma.assign(model.model_down.gamma + gamma_rate)

    train_scores = np.zeros(ROUNDS)
    for i in range(ROUNDS):
        # Randomize the game environment for each batch
        games.randomize_environment_all()
        
        # Create training batch
        o0, o1, pi0, log_Ppi = u.make_batch_dsprites_active_inference(
            games=games,
            model=model.active_inf_model,
            deepness=deepness,
            samples=samples,
            calc_mean=True,
            repeats=repeats
        )

        # Train top layer
        qs0, _, _ = model.model_down.encoder_with_sample(o0)
        D_KL_pi = loss.train_model_top(
            model_top=model.model_top,
            s=qs0,
            log_Ppi=log_Ppi,
            optimizer=optimizers['top']
        )
        D_KL_pi = D_KL_pi.numpy()

        current_omega = loss.compute_omega(D_KL_pi, a=var_a, b=var_b, c=var_c, d=var_d).reshape(-1, 1)

        # Train middle layer
        qs1_mean, qs1_logvar = model.model_down.encoder(o1)
        ps1_mean, ps1_logvar = loss.train_model_mid(
            model_mid=model.model_mid,
            s0=qs0,
            qs1_mean=qs1_mean,
            qs1_logvar=qs1_logvar,
            Ppi_sampled=pi0,
            omega=current_omega,
            optimizer=optimizers['mid']
        )

        # Train bottom layer
        loss.train_model_down(
            model_down=model.model_down,
            o1=o1,
            ps1_mean=ps1_mean,
            ps1_logvar=ps1_logvar,
            omega=current_omega,
            optimizer=optimizers['down']
        )

    # Save the model and stats every 2 epochs
    if epoch % 2 == 0:
        model.save_all(folder_chp, stats, script_file=argv[0], optimizers=optimizers)
    if epoch % 25 == 0:
        # Keep checkpoints every 25 epochs
        copy_tree(folder_chp, f"{folder_chp}_epoch_{epoch}")

    # Generate test batch
    o0, o1, pi0, S0_real, _ = u.make_batch_dsprites_random(
        game=game_test,
        index=0,
        size=TEST_SIZE,
        repeats=repeats
    )
    
    # Compute top layer loss
    log_Ppi = np.log(pi0 + 1e-15)
    s0, _, _ = model.model_down.encoder_with_sample(o0)
    F_top, kl_div_pi, kl_div_pi_anal, Qpi = loss.compute_loss_top(
        model_top=model.model_top,
        s=s0,
        log_Ppi=log_Ppi
    )

    # Compute middle layer loss
    qs1_mean, qs1_logvar = model.model_down.encoder(o1)
    qs1 = model.model_down.reparameterize(qs1_mean, qs1_logvar)
    F_mid, loss_terms_mid, ps1, ps1_mean, ps1_logvar = loss.compute_loss_mid(
        model_mid=model.model_mid,
        s0=s0,
        Ppi_sampled=pi0,
        qs1_mean=qs1_mean,
        qs1_logvar=qs1_logvar,
        omega=var_a / 2.0 + var_d
    )

    # Compute bottom layer loss
    F_down, loss_terms, po1, qs1 = loss.compute_loss_down(
        model_down=model.model_down,
        o1=o1,
        ps1_mean=ps1_mean,
        ps1_logvar=ps1_logvar,
        omega=var_a / 2.0 + var_d
    )

    # Update stats
    stats['F'].append(np.mean(F_down) + np.mean(F_mid) + np.mean(F_top))
    stats['F_top'].append(np.mean(F_top))
    stats['F_mid'].append(np.mean(F_mid))
    stats['F_down'].append(np.mean(F_down))
    stats['mse_o'].append(np.mean(loss_terms[0]))
    stats['kl_div_s'].append(np.mean(loss_terms[1]))
    stats['kl_div_s_anal'].append(np.mean(loss_terms[2], axis=0))
    stats['omega'].append(np.mean(current_omega))
    stats['omega_std'].append(np.std(current_omega))
    stats['kl_div_pi'].append(np.mean(kl_div_pi))
    stats['TC'].append(np.mean(total_correlation(qs1.numpy())))

    # Visualize results
    generate_traversals(
        model=model.active_inf_model,
        s_dim=s_dim,
        s_sample=s0,
        S_real=S0_real,
        filenames=[f"{folder}/traversals_at_epoch_{epoch:04d}.png"]
    )
    reconstructions_plot(o0, o1, po1.numpy(), filename=f"{folder}/imagination_{signature}_{epoch}.png")

    print(f"{epoch}, F: {stats['F'][-1]:.2f}, MSEo: {stats['mse_o'][-1]:.3f}, "
          f"KLs: {stats['kl_div_s'][-1]:.2f}, omega: {stats['omega'][-1]:.2f}, "
          f"TC: {stats['TC'][-1]:.2f}, dur. {round(time.time()-start_time, 2)}s")
    start_time = time.time()
