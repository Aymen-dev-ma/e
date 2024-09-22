# comparison+.py

import os
import numpy as np
import tensorflow as tf
from src.tfmodel import ActiveInferenceModel, CausalInferenceModel
from src.game_environment import Game
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings for cleaner output
import logging
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Define helper function to create one-hot encoded vectors
def create_one_hot(action, pi_dim):
    """
    Converts an integer action into a one-hot encoded float32 vector.

    Args:
        action (int): Action index.
        pi_dim (int): Dimension of the action space.

    Returns:
        np.ndarray: One-hot encoded action vector of shape (pi_dim,).
    """
    one_hot = np.zeros(pi_dim, dtype=np.float32)
    one_hot[action] = 1.0
    return one_hot

# Define helper function to calculate moving average
def moving_average(data, window_size):
    """
    Calculates the moving average of a 1D array.

    Args:
        data (list or np.ndarray): Input data.
        window_size (int): Size of the moving window.

    Returns:
        np.ndarray: Moving average of the input data.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():
    # Paths to the saved models
    active_model_path = '/content/deep-active-inference-mc/figs_final_model_0.01_30_1.0_50_10_5/checkpoints'
    causal_model_path = '/content/deep-active-inference-mc/figs_causal_model_0.01_30_1.0_50_10_5/checkpoints'

    # Define the directory where the comparison plot will be saved
    comparison_save_dir = '/content/deep-active-inference-mc/figs_causal_model_0.01_30_1.0_50_10_5'
    os.makedirs(comparison_save_dir, exist_ok=True)
    comparison_plot_path = os.path.join(comparison_save_dir, 'active_vs_causal_scores.png')

    # Load the Active Inference Model
    s_dim = 10
    pi_dim = 4
    active_model = ActiveInferenceModel(
        s_dim=s_dim,
        pi_dim=pi_dim,
        gamma=1.0,
        beta_s=1.0,
        beta_o=1.0,
        colour_channels=1,
        resolution=64
    )
    active_model.load_all(active_model_path)
    print("Active Inference Model loaded successfully.")

    # Load the Causal Inference Model
    causal_model = CausalInferenceModel(
        s_dim=s_dim,
        pi_dim=pi_dim,
        gamma=1.0,
        beta_s=1.0,
        beta_o=1.0,
        colour_channels=1,
        resolution=64
    )
    causal_model.load_all(causal_model_path)
    print("Causal Inference Model loaded successfully.")

    # Initialize the game environment
    game = Game(1)

    # Number of test rounds to compare
    TEST_ROUNDS = 5000  # Increased from 100 to 1000 for less volatility

    # Store scores for comparison
    active_scores = []
    causal_scores = []

    # Define moving average window size
    MA_WINDOW = 50

    # Test Active Inference Model
    print("\nTesting Active Inference Model")
    for round_num in range(1, TEST_ROUNDS + 1):
        # Reset or randomize the environment
        game.randomize_environment(0)
        o0 = game.current_frame(0).reshape(1, 64, 64, 1).astype(np.float32)  # Ensure float32

        # Generate a dummy action (randomly for now)
        action_index = np.random.choice(pi_dim, p=[0.25, 0.25, 0.25, 0.25])
        pi0_one_hot = create_one_hot(action_index, pi_dim)

        # Convert to TensorFlow tensor
        pi0_tensor = tf.convert_to_tensor(pi0_one_hot.reshape(1, pi_dim), dtype=tf.float32)

        # Predict future observation using Active Inference model
        try:
            future_o = active_model.imagine_future_from_o(tf.convert_to_tensor(o0), pi0_tensor)
        except Exception as e:
            print(f"Error during Active Inference prediction at round {round_num}: {e}")
            future_o = None

        # Simulate the game to get the reward
        try:
            reward = game.get_reward(0)
            active_scores.append(reward)
        except Exception as e:
            print(f"Error retrieving reward for Active Inference at round {round_num}: {e}")
            active_scores.append(0)  # Append zero or handle as appropriate

        if round_num % 100 == 0:
            print(f"Active Inference Model: Completed {round_num}/{TEST_ROUNDS} rounds.")

    # Test Causal Inference Model
    print("\nTesting Causal Inference Model")
    for round_num in range(1, TEST_ROUNDS + 1):
        # Reset or randomize the environment
        game.randomize_environment(0)
        o0 = game.current_frame(0).reshape(1, 64, 64, 1).astype(np.float32)  # Ensure float32

        # Generate a dummy action (randomly for now)
        action_index = np.random.choice(pi_dim, p=[0.25, 0.25, 0.25, 0.25])
        pi0_one_hot = create_one_hot(action_index, pi_dim)

        # Convert to TensorFlow tensor
        pi0_tensor = tf.convert_to_tensor(pi0_one_hot.reshape(1, pi_dim), dtype=tf.float32)

        # Predict future observation using Causal Inference model
        try:
            future_o = causal_model.imagine_future_from_o(tf.convert_to_tensor(o0), pi0_tensor)
        except Exception as e:
            print(f"Error during Causal Inference prediction at round {round_num}: {e}")
            future_o = None

        # Simulate the game to get the reward
        try:
            reward = game.get_reward(0)
            causal_scores.append(reward)
        except Exception as e:
            print(f"Error retrieving reward for Causal Inference at round {round_num}: {e}")
            causal_scores.append(0)  # Append zero or handle as appropriate

        if round_num % 100 == 0:
            print(f"Causal Inference Model: Completed {round_num}/{TEST_ROUNDS} rounds.")

    # Calculate moving averages
    active_moving_avg = moving_average(active_scores, MA_WINDOW)
    causal_moving_avg = moving_average(causal_scores, MA_WINDOW)

    # Compare results
    active_mean_score = np.mean(active_scores)
    causal_mean_score = np.mean(causal_scores)

    print(f"\nActive Inference Model average score: {active_mean_score}")
    print(f"Causal Inference Model average score: {causal_mean_score}")

    # Plot the scores with moving averages
    plt.figure(figsize=(14, 7))
    
    # Plot individual scores with transparency
    plt.scatter(range(TEST_ROUNDS), active_scores, label="Active Inference Scores", color='blue', alpha=0.1)
    plt.scatter(range(TEST_ROUNDS), causal_scores, label="Causal Inference Scores", color='red', alpha=0.1)
    
    # Plot moving averages
    plt.plot(range(MA_WINDOW -1, TEST_ROUNDS), active_moving_avg, label=f"Active Inference MA ({MA_WINDOW})", color='blue')
    plt.plot(range(MA_WINDOW -1, TEST_ROUNDS), causal_moving_avg, label=f"Causal Inference MA ({MA_WINDOW})", color='red')
    
    plt.xlabel("Test Round")
    plt.ylabel("Score")
    plt.title("Active vs Causal Inference Model Scores with Moving Averages")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the comparison plot
    plt.savefig(comparison_plot_path)
    print(f"Comparison plot saved to {comparison_plot_path}")

    # Optionally, display the plot
    plt.show()

if __name__ == "__main__":
    main()
