import tensorflow as tf
import pickle
import numpy as np
from shutil import copyfile
from src.tfutils import *

class ModelTop(tf.keras.Model):
    def __init__(self, s_dim, pi_dim, tf_precision, precision):
        super(ModelTop, self).__init__()
        # For activation function we used ReLU.
        # For weight initialization we used He Uniform

        self.tf_precision = tf_precision
        self.precision = precision
        tf.keras.backend.set_floatx(self.precision)

        self.s_dim = s_dim
        self.pi_dim = pi_dim
 
        self.qpi_net = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape=(s_dim,)),
              tf.keras.layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dense(pi_dim),]) # No activation

    def encode_s(self, s0):
        logits_pi = self.qpi_net(s0)
        q_pi = tf.nn.softmax(logits_pi)
        log_q_pi = tf.math.log(q_pi+1e-20)
        return logits_pi, q_pi, log_q_pi

class ModelMid(tf.keras.Model):
    def __init__(self, s_dim, pi_dim, tf_precision, precision):
        super(ModelMid, self).__init__()

        self.tf_precision = tf_precision
        self.precision = precision
        tf.keras.backend.set_floatx(self.precision)

        self.s_dim = s_dim
        self.pi_dim = pi_dim

        self.ps_net = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape=(pi_dim+s_dim,)),
              tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(s_dim + s_dim),]) # No activation

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def transition(self, pi, s0):
        mean, logvar = tf.split(self.ps_net(tf.concat([pi,s0],1)), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def transition_with_sample(self, pi, s0):
        ps1_mean, ps1_logvar = self.transition(pi, s0)
        ps1 = self.reparameterize(ps1_mean, ps1_logvar)
        return ps1, ps1_mean, ps1_logvar

class ModelDown(tf.keras.Model):
    def __init__(self, s_dim, pi_dim, tf_precision, precision, colour_channels, resolution):
        super(ModelDown, self).__init__()

        self.tf_precision = tf_precision
        self.precision = precision
        tf.keras.backend.set_floatx(self.precision)

        self.s_dim = s_dim
        self.pi_dim = pi_dim
        self.colour_channels = colour_channels
        self.resolution = resolution
        if self.resolution == 64:
            last_strides = 2
        elif self.resolution == 32:
            last_strides = 1
        else:
            exit('Unknown resolution..')

        self.qs_net = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape=(self.resolution, self.resolution, self.colour_channels)),
              tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', kernel_initializer='he_uniform'),
              tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', kernel_initializer='he_uniform'),
              tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', kernel_initializer='he_uniform'),
              tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', kernel_initializer='he_uniform'),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(s_dim + s_dim),]) # No activation
        self.po_net = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape=(s_dim,)),
              tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(units=16*16*64, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Reshape(target_shape=(16, 16, 64)),
              tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(1, 1), padding="SAME", activation='relu', kernel_initializer='he_uniform'),
              tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu', kernel_initializer='he_uniform'),
              tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(last_strides, last_strides), padding="SAME", activation='relu', kernel_initializer='he_uniform'),
              tf.keras.layers.Conv2DTranspose(filters=self.colour_channels, kernel_size=3, strides=(1, 1), padding="SAME", activation='sigmoid', kernel_initializer='he_uniform'),])

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def encoder(self, o):
        grad_check = tf.debugging.check_numerics(self.qs_net(o), 'check_numerics caught bad temptemptemptemptemp3')
        mean_s, logvar_s = tf.split(self.qs_net(o), num_or_size_splits=2, axis=1)
        return mean_s, logvar_s

    @tf.function
    def decoder(self, s):
        po = self.po_net(s)
        return po

    @tf.function
    def encoder_with_sample(self, o):
        mean, logvar = self.encoder(o)
        s = self.reparameterize(mean, logvar)
        return s, mean, logvar
######################################################
import tensorflow as tf
import numpy as np
import logging
import pickle
import networkx as nx
from sklearn.linear_model import LinearRegression
from shutil import copyfile

class AdvancedCausalInferenceModel(tf.Module):
    """
    Implements advanced causal inference techniques like front-door and back-door adjustments.
    Enhances the decision-making process with precise causal effect estimation.
    """
    def __init__(self, s_dim, pi_dim, gamma, beta_s, beta_o, colour_channels=1, resolution=64):
        """
        Initializes the AdvancedCausalInferenceModel with dynamic causal graph and inference models.
        """
        super(AdvancedCausalInferenceModel, self).__init__()
        self.s_dim = s_dim
        self.pi_dim = pi_dim
        self.gamma = gamma
        self.beta_s = beta_s
        self.beta_o = beta_o
        self.colour_channels = colour_channels
        self.resolution = resolution

        # Initialize Active Inference components
        self.active_inf_model = ActiveInferenceModel(
            s_dim=self.s_dim,
            pi_dim=self.pi_dim,
            gamma=self.gamma,
            beta_s=self.beta_s,
            beta_o=self.beta_o,
            colour_channels=self.colour_channels,
            resolution=self.resolution
        )

        # Directly expose model components
        self.model_down = self.active_inf_model.model_down
        self.model_mid = self.active_inf_model.model_mid
        self.model_top = self.active_inf_model.model_top

        # Initialize causal graph
        self.causal_graph = nx.DiGraph()
        self._initialize_causal_graph()

        # Store mediator and outcome data for front-door adjustment
        self.mediator_data = []
        self.outcome_data = []

        # Set up logging
        self.logger = logging.getLogger('AdvancedCausalInferenceModel')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def _initialize_causal_graph(self):
        """
        Initialize nodes for actions (pi) and states (s) in the causal graph.
        """
        for pi in range(self.pi_dim):
            self.causal_graph.add_node(f'pi_{pi}', type='action')
        for s in range(self.s_dim):
            self.causal_graph.add_node(f's_{s}', type='state')

    def update_causal_graph(self, s0, pi, s1):
        """
        Updates the causal graph when observing state transitions under an action.
        """
        changed_dims = np.where(~np.isclose(s0, s1, atol=1e-3))[0]
        action_index = np.argmax(pi)

        for dim in changed_dims:
            action_node = f'pi_{action_index}'
            state_node = f's_{dim}'
            if self.causal_graph.has_edge(action_node, state_node):
                self.causal_graph[action_node][state_node]['weight'] += 1.0
            else:
                self.causal_graph.add_edge(action_node, state_node, weight=1.0)

        self.logger.debug(f"Causal graph updated with action pi_{action_index} affecting state(s) {changed_dims}")

    def collect_frontdoor_data(self, pi, s1, o1):
        """
        Collects mediator and outcome data for front-door adjustment.
        Limits the size of the collected data to avoid memory issues.
        """
        mediator = s1
        outcome = o1.flatten()

        self.mediator_data.append(mediator)
        self.outcome_data.append(outcome)

        if len(self.mediator_data) > 5000:  # Limit the data size
            self.mediator_data = self.mediator_data[-5000:]
            self.outcome_data = self.outcome_data[-5000:]

    def perform_backdoor_adjustment(self, pi):
        """
        Performs backdoor adjustment to estimate causal effects using known dependencies in the graph.
        """
        action_index = np.argmax(pi)
        action_node = f'pi_{action_index}'
        affected_states = list(self.causal_graph.successors(action_node))

        # Implement dynamic conditional independence testing or causal effect estimation here
        if affected_states:
            self.logger.info(f"Backdoor adjustment: Found effects for action pi_{action_index} on state(s) {affected_states}")
        else:
            self.logger.warning(f"Backdoor adjustment: No significant causal effect found for action pi_{action_index}")

        # Return an effect score (this can be expanded with actual backdoor criteria)
        return len(affected_states)

    def perform_frontdoor_adjustment(self):
        """
        Performs front-door adjustment to estimate causal effects using mediator-outcome models.
        """
        if len(self.mediator_data) < 100:
            self.logger.warning("Not enough data to perform front-door adjustment.")
            return 0

        # Fit mediator-to-outcome model
        mediators = np.array(self.mediator_data)
        outcomes = np.array(self.outcome_data)
        outcomes_flat = outcomes.reshape(outcomes.shape[0], -1)

        mediator_to_outcome_model = LinearRegression()
        mediator_to_outcome_model.fit(mediators, outcomes_flat)

        self.logger.info("Front-door adjustment performed and causal effect estimated.")
        return mediator_to_outcome_model  # This can be stored or used for causal effect prediction

    def estimate_causal_effect(self, action):
        """
        Estimates the causal effect of a given action using front-door and backdoor adjustments.
        """
        backdoor_effect = self.perform_backdoor_adjustment(action)

        # Use front-door adjustment if mediator data is available
        if len(self.mediator_data) > 100:
            frontdoor_model = self.perform_frontdoor_adjustment()
            if frontdoor_model:
                mediator_estimate = np.mean(frontdoor_model.predict(self.mediator_data))
                self.logger.info(f"Estimated causal effect using front-door adjustment: {mediator_estimate}")
                return mediator_estimate + backdoor_effect
        return backdoor_effect

    def choose_action(self, o0):
        """
        Chooses the best action using both habitual network predictions and causal effect estimations.
        """
        best_action = None
        best_effect = -np.inf

        for a_index in range(self.pi_dim):
            action = np.zeros(self.pi_dim)
            action[a_index] = 1.0

            causal_effect = self.estimate_causal_effect(action)
            if causal_effect > best_effect:
                best_effect = causal_effect
                best_action = action

        return best_action if best_action is not None else self.habitual_net(o0)

    def habitual_net(self, o):
        """
        Uses the habitual network to predict action probabilities.
        """
        qs_mean, _ = self.model_down.encoder(o)
        _, Qpi, _ = self.model_top.encode_s(qs_mean)
        return Qpi

    def update_on_transition(self, o0, pi, o1):
        """
        Updates the causal graph and collects data for front-door adjustment after observing a transition.
        """
        s0_mean, _ = self.model_down.encoder(o0)
        s1_mean, _ = self.model_down.encoder(o1)

        self.update_causal_graph(s0_mean.numpy().flatten(), pi.numpy().flatten(), s1_mean.numpy().flatten())
        self.collect_frontdoor_data(pi.numpy(), s1_mean.numpy(), o1.numpy())

    def save_all(self, folder_chp, stats, script_file="", optimizers={}):
        """
        Saves model weights, causal graph, and statistics.
        """
        self.active_inf_model.save_weights(folder_chp)
        self.save_causal_graph(folder_chp + '/causal_graph.pkl')

        with open(folder_chp + '/stats.pkl', 'wb') as f:
            pickle.dump(stats, f)

        for key, optimizer in optimizers.items():
            checkpoint = tf.train.Checkpoint(optimizer=optimizer)
            checkpoint.save(file_prefix=folder_chp + f'/optimizer_{key}')

        if script_file != "":
            copyfile(script_file, folder_chp + '/' + script_file)

    def load_all(self, folder_chp):
        """
        Loads model weights, causal graph, and statistics.
        """
        self.active_inf_model.load_weights(folder_chp)
        self.load_causal_graph(folder_chp + '/causal_graph.pkl')

        with open(folder_chp + '/stats.pkl', 'rb') as f:
            stats = pickle.load(f)

        optimizers = {}
        for key in ['down', 'mid', 'top']:
            optimizer = tf.keras.optimizers.Adam()
            checkpoint = tf.train.Checkpoint(optimizer=optimizer)
            checkpoint.restore(tf.train.latest_checkpoint(folder_chp))
            optimizers[key] = optimizer

        return stats, optimizers

    def save_causal_graph(self, filepath):
        """
        Saves the causal graph to a file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.causal_graph, f)
        self.logger.info(f"Causal graph saved to {filepath}")

    def load_causal_graph(self, filepath):
        """
        Loads the causal graph from a file.
        """
        with open(filepath, 'rb') as f:
            self.causal_graph = pickle.load(f)
        self.logger.info(f"Causal graph loaded from {filepath}")

#####################################################

import tensorflow as tf
import numpy as np
import logging
import pickle
import networkx as nx
from sklearn.linear_model import LinearRegression
from shutil import copyfile
class CausalInferenceModel(tf.Module):
    """
    Integrates advanced causal reasoning into the Active Inference framework.
    It builds and maintains a causal graph based on the agent's interactions
    with the environment and uses it to enhance decision-making processes
    through techniques like front-door adjustment.
    """
    def __init__(self, s_dim, pi_dim, gamma, beta_s, beta_o, colour_channels=1, resolution=64):
        """
        Initializes the CausalInferenceModel by initializing the ActiveInferenceModel
        and setting up the advanced causal reasoning components.
        """
        super(CausalInferenceModel, self).__init__()
        self.s_dim = s_dim
        self.pi_dim = pi_dim
        self.gamma = gamma
        self.beta_s = beta_s
        self.beta_o = beta_o
        self.colour_channels = colour_channels
        self.resolution = resolution

        # Initialize ActiveInferenceModel
        self.active_inf_model = ActiveInferenceModel(
            s_dim=self.s_dim,
            pi_dim=self.pi_dim,
            gamma=self.gamma,
            beta_s=self.beta_s,
            beta_o=self.beta_o,
            colour_channels=self.colour_channels,
            resolution=self.resolution
        )

        # Expose model components directly for convenience
        self.model_down = self.active_inf_model.model_down
        self.model_mid = self.active_inf_model.model_mid
        self.model_top = self.active_inf_model.model_top

        # Initialize the Causal Graph
        self.causal_graph = nx.DiGraph()
        self._initialize_causal_graph()

        # Parameters for causal adjustments
        self.causal_boost = 0.1    # Boost value for beneficial actions
        self.causal_penalty = 0.1  # Penalty value for detrimental actions

        # Data storage for front-door adjustment
        self.frontdoor_data = []

        # Models for front-door estimation
        self.action_to_mediator_model = None
        self.mediator_to_outcome_model = None

        # Logging setup
        self.logger = logging.getLogger('CausalInferenceModel')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def _initialize_causal_graph(self):
        """
        Initializes the causal graph nodes based on state and action dimensions.
        """
        # Add action nodes
        for pi in range(self.pi_dim):
            self.causal_graph.add_node(f'pi_{pi}', type='action')

        # Add state nodes
        for s in range(self.s_dim):
            self.causal_graph.add_node(f's_{s}', type='state')

    def update_causal_graph(self, s0, pi, s1):
        """
        Updates the causal graph based on a state transition influenced by an action.
        """
        # Identify changed state dimensions
        changed_dims = np.where(~np.isclose(s0, s1, atol=1e-3))[0]
        action_index = np.argmax(pi)

        for dim in changed_dims:
            action_node = f'pi_{action_index}'
            state_node = f's_{dim}'
            if self.causal_graph.has_edge(action_node, state_node):
                # If edge exists, increment weight
                self.causal_graph[action_node][state_node]['weight'] += 1.0
            else:
                # Add edge with initial weight
                self.causal_graph.add_edge(action_node, state_node, weight=1.0)

        self.logger.debug(f"Updated causal graph with action pi_{action_index} affecting states {changed_dims}")

    def collect_frontdoor_data(self, pi, s1, o1):
        """
        Collects data required for front-door adjustment.
        Limits the size of the collected data to prevent memory issues.
        """
        batch_size = pi.shape[0]
        for i in range(batch_size):
            action = pi[i]
            mediator = s1[i]
            outcome = o1[i].flatten()
            self.frontdoor_data.append({
                'action': action,
                'mediator': mediator,
                'outcome': outcome
            })
        # Limit the size of frontdoor_data to prevent memory issues
        max_data_size = 5000  # Adjust as needed
        if len(self.frontdoor_data) > max_data_size:
            self.frontdoor_data = self.frontdoor_data[-max_data_size:]

    def perform_frontdoor_adjustment(self):
        """
        Performs front-door adjustment to estimate causal effects.
        """
        if not self.frontdoor_data:
            return  # No data to perform adjustment

        # Prepare data
        actions = np.array([data['action'] for data in self.frontdoor_data])
        mediators = np.array([data['mediator'] for data in self.frontdoor_data])
        outcomes = np.array([data['outcome'] for data in self.frontdoor_data])

        # Flatten actions to indices
        action_indices = np.argmax(actions, axis=1)

        # Step 1: Estimate P(M | A)
        self.action_to_mediator_model = []
        for dim in range(self.s_dim):
            model = LinearRegression()
            model.fit(action_indices.reshape(-1, 1), mediators[:, dim])
            self.action_to_mediator_model.append(model)

        # Step 2: Estimate P(Y | M)
        self.mediator_to_outcome_model = LinearRegression()
        mediators_flat = mediators.reshape(mediators.shape[0], -1)
        outcomes_flat = outcomes.reshape(outcomes.shape[0], -1)
        self.mediator_to_outcome_model.fit(mediators_flat, outcomes_flat)

        self.logger.info("Performed front-door adjustment and updated causal effect estimations.")

    def estimate_causal_effect(self, action):
        """
        Estimates the causal effect of a specific action using the front-door adjustment.
        """
        if self.action_to_mediator_model is None or self.mediator_to_outcome_model is None:
            return 0  # Models are not trained yet

        action_index = np.argmax(action)
        mediator_estimates = []
        for dim, model in enumerate(self.action_to_mediator_model):
            mediator_estimate = model.predict([[action_index]])[0]
            mediator_estimates.append(mediator_estimate)
        mediator_estimates = np.array(mediator_estimates).reshape(1, -1)
        outcome_estimate = self.mediator_to_outcome_model.predict(mediator_estimates)[0]
        causal_effect = np.mean(outcome_estimate)
        return causal_effect

    def choose_action_with_frontdoor(self, o0):
        """
        Chooses the best action based on front-door adjustment.
        """
        best_action = None
        best_effect = -np.inf
        for a_index in range(self.pi_dim):
            action = np.zeros(self.pi_dim)
            action[a_index] = 1.0
            causal_effect = self.estimate_causal_effect(action)
            if causal_effect > best_effect:
                best_effect = causal_effect
                best_action = action
        if best_action is not None:
            return best_action
        else:
            # Fallback to habitual network if no causal effect is estimated
            Qpi = self.habitual_net(o0).numpy()
            action_index = np.argmax(Qpi)
            action = np.zeros(self.pi_dim)
            action[action_index] = 1.0
            return action

    def integrate_causal_reasoning(self, o0):
        """
        Integrates causal reasoning into the Active Inference process by selecting
        actions based on the estimated causal effects using front-door adjustment.
        """
        # Choose action using front-door adjustment
        action = self.choose_action_with_frontdoor(o0)
        return action

    def update_on_transition(self, o0, pi, o1):
        """
        Updates the causal graph and collects data for front-door adjustment
        based on the observed transition.
        """
        # Encode initial and resulting observations to obtain state representations
        s0_mean, _ = self.model_down.encoder(o0)
        s1_mean, _ = self.model_down.encoder(o1)
        s0_np = s0_mean.numpy().flatten()
        s1_np = s1_mean.numpy().flatten()
        pi_np = pi

        # Update causal graph with the observed transition
        self.update_causal_graph(s0_np, pi_np, s1_np)

        # Collect data for front-door adjustment
        self.collect_frontdoor_data(pi_np[np.newaxis, :], s1_mean.numpy(), o1.numpy())

    def save_weights(self, folder_chp):
        """
        Saves the weights of the ActiveInferenceModel and the causal graph.
        """
        self.active_inf_model.save_weights(folder_chp)
        self.save_causal_graph(folder_chp + '/causal_graph.pkl')

    def load_weights(self, folder_chp):
        """
        Loads the weights of the ActiveInferenceModel and the causal graph.
        """
        self.active_inf_model.load_weights(folder_chp)
        self.load_causal_graph(folder_chp + '/causal_graph.pkl')

    def save_causal_graph(self, filepath):
        """
        Saves the causal graph to a file using pickle.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.causal_graph, f)
        self.logger.info(f"Causal graph saved to {filepath}")

    def load_causal_graph(self, filepath):
        """
        Loads the causal graph from a file using pickle.
        """
        with open(filepath, 'rb') as f:
            self.causal_graph = pickle.load(f)
        self.logger.info(f"Causal graph loaded from {filepath}")

    def save_all(self, folder_chp, stats, script_file="", optimizers={}):
        """
        Saves all components including weights, stats, optimizer states, and source files.
        """
        self.save_weights(folder_chp)

        # Save stats
        with open(folder_chp + '/stats.pkl', 'wb') as ff:
            pickle.dump(stats, ff)

        # Save optimizer state using Checkpoint API
        for key, optimizer in optimizers.items():
            checkpoint = tf.train.Checkpoint(optimizer=optimizer)
            checkpoint.save(file_prefix=folder_chp + f'/optimizer_{key}')

        # Copy source files for reference
        copyfile('src/tfmodel.py', folder_chp + '/tfmodel.py')
        copyfile('src/tfloss.py', folder_chp + '/tfloss.py')
        if script_file != "":
            copyfile(script_file, folder_chp + '/' + script_file)

    def load_all(self, folder_chp):
        """
        Loads all components including weights, stats, optimizer states, and source files.
        """
        self.load_weights(folder_chp)

        # Load stats
        with open(folder_chp + '/stats.pkl', 'rb') as ff:
            stats = pickle.load(ff)

        optimizers = {}

        # Restore optimizer state using Checkpoint API
        for key in ['down', 'mid', 'top']:
            optimizer = tf.keras.optimizers.Adam()
            checkpoint = tf.train.Checkpoint(optimizer=optimizer)
            latest_checkpoint = tf.train.latest_checkpoint(folder_chp)
            if latest_checkpoint:
                checkpoint.restore(latest_checkpoint)
            optimizers[key] = optimizer

        # Restore model parameters
        if 'var_beta_s' in stats and len(stats['var_beta_s']) > 0:
            self.model_down.beta_s.assign(stats['var_beta_s'][-1])
        if 'var_gamma' in stats and len(stats['var_gamma']) > 0:
            self.model_down.gamma.assign(stats['var_gamma'][-1])
        if 'var_beta_o' in stats and len(stats['var_beta_o']) > 0:
            self.model_down.beta_o.assign(stats['var_beta_o'][-1])

        return stats, optimizers

    def habitual_net(self, o):
        """
        Computes the habitual network's action probabilities.
        """
        Qpi = self.active_inf_model.habitual_net(o)
        return Qpi

    def imagine_future_from_o(self, o0, pi):
        """
        Imagines the future observation based on the current observation and action.
        """
        return self.active_inf_model.imagine_future_from_o(o0, pi)

    def calculate_G_repeated(self, o, pi, steps=1, calc_mean=False, samples=10):
        """
        Calculates the expected free energy (G) for repeated actions.
        """
        return self.active_inf_model.calculate_G_repeated(o, pi, steps, calc_mean, samples)

    def calculate_G_4_repeated(self, o, steps=1, calc_mean=False, samples=10):
        """
        Calculates the expected free energy (G) for four repeated actions.
        """
        return self.active_inf_model.calculate_G_4_repeated(o, steps, calc_mean, samples)

    def calculate_G(self, s0, pi0, samples=10):
        """
        Calculates the expected free energy (G) for given states and actions.
        """
        return self.active_inf_model.calculate_G(s0, pi0, samples)

    def calculate_G_mean(self, s0, pi0):
        """
        Calculates the expected free energy (G) using mean states.
        """
        return self.active_inf_model.calculate_G_mean(s0, pi0)

    def calculate_G_given_trajectory(self, s0_traj, ps1_traj, ps1_mean_traj, ps1_logvar_traj, pi0_traj):
        """
        Calculates the expected free energy (G) given a trajectory.
        """
        return self.active_inf_model.calculate_G_given_trajectory(s0_traj, ps1_traj, ps1_mean_traj, ps1_logvar_traj, pi0_traj)

    def check_reward(self, o):
        """
        Computes the reward based on the observation.
        """
        if self.active_inf_model.model_down.resolution == 64:
            return tf.reduce_mean(calc_reward(o), axis=[1, 2, 3]) * 10.0
        elif self.active_inf_model.model_down.resolution == 32:
            return tf.reduce_sum(calc_reward_animalai(o), axis=[1, 2, 3])
#####################################################

import tensorflow as tf
import numpy as np
import logging
import pickle
import networkx as nx
from shutil import copyfile
class CausalInferenceModel(tf.Module):
    """
    This class integrates causal reasoning into the Active Inference framework.
    It builds and maintains a causal graph based on the agent's interactions
    with the environment and uses it to enhance decision-making processes.
    """
    def __init__(self, s_dim, pi_dim, gamma, beta_s, beta_o, colour_channels=1, resolution=64):
        """
        Initializes the CausalInferenceModel by initializing the ActiveInferenceModel
        and setting up the causal reasoning components.
        """
        super(CausalInferenceModel, self).__init__()
        self.s_dim = s_dim
        self.pi_dim = pi_dim
        self.gamma = gamma
        self.beta_s = beta_s
        self.beta_o = beta_o
        self.colour_channels = colour_channels
        self.resolution = resolution

        # Initialize ActiveInferenceModel
        self.active_inf_model = ActiveInferenceModel(
            s_dim=self.s_dim,
            pi_dim=self.pi_dim,
            gamma=self.gamma,
            beta_s=self.beta_s,
            beta_o=self.beta_o,
            colour_channels=self.colour_channels,
            resolution=self.resolution
        )

        # Expose model components directly for convenience
        self.model_down = self.active_inf_model.model_down
        self.model_mid = self.active_inf_model.model_mid
        self.model_top = self.active_inf_model.model_top

        # Initialize the Causal Graph
        self.causal_graph = nx.DiGraph()
        self._initialize_causal_graph()

        # Parameters for causal adjustments
        self.causal_boost = 0.1    # Boost value for beneficial actions
        self.causal_penalty = 0.1  # Penalty value for detrimental actions

        # Logging setup
        self.logger = logging.getLogger('CausalInferenceModel')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def _initialize_causal_graph(self):
        """
        Initializes the causal graph nodes based on state and action dimensions.
        """
        # Add action nodes
        for pi in range(self.pi_dim):
            self.causal_graph.add_node(f'pi_{pi}', type='action')

        # Add state nodes
        for s in range(self.s_dim):
            self.causal_graph.add_node(f's_{s}', type='state')

    def update_causal_graph(self, s0, pi, s1):
        """
        Updates the causal graph based on a state transition influenced by an action.
        """
        # Identify changed state dimensions
        changed_dims = np.where(~np.isclose(s0, s1, atol=1e-3))[0]
        action_index = np.argmax(pi)

        for dim in changed_dims:
            action_node = f'pi_{action_index}'
            state_node = f's_{dim}'
            if self.causal_graph.has_edge(action_node, state_node):
                # If edge exists, increment weight
                self.causal_graph[action_node][state_node]['weight'] += 1.0
            else:
                # Add edge with initial weight
                self.causal_graph.add_edge(action_node, state_node, weight=1.0)

        self.logger.debug(f"Updated causal graph with action pi_{action_index} affecting states {changed_dims}")

    def get_causal_effects(self, pi):
        """
        Retrieves the causal effects of a given action.
        """
        action_index = np.argmax(pi)
        action_node = f'pi_{action_index}'
        if self.causal_graph.has_node(action_node):
            affected_states = list(self.causal_graph.successors(action_node))
            return affected_states
        return []

    def adjust_policy_with_causality(self, Qpi):
        """
        Adjusts the policy probabilities based on causal effects.
        """
        Qpi_np = Qpi.numpy().copy()
        adjusted_Qpi = Qpi_np.copy()

        for i in range(Qpi_np.shape[0]):
            for pi in range(self.pi_dim):
                pi_one_hot = np.zeros(self.pi_dim, dtype=np.float32)
                pi_one_hot[pi] = 1.0
                effects = self.get_causal_effects(pi_one_hot)
                if effects:
                    # Boost probability for actions with known causal effects
                    adjusted_Qpi[i, pi] += self.causal_boost
                else:
                    # Penalize actions with no known causal effects
                    adjusted_Qpi[i, pi] -= self.causal_penalty

            # Ensure probabilities are non-negative
            adjusted_Qpi[i] = np.clip(adjusted_Qpi[i], 0, None)
            # Normalize to sum to 1
            if adjusted_Qpi[i].sum() > 0:
                adjusted_Qpi[i] /= adjusted_Qpi[i].sum()
            else:
                adjusted_Qpi[i] = Qpi_np[i]

        self.logger.debug("Adjusted policy probabilities based on causal reasoning")
        return tf.convert_to_tensor(adjusted_Qpi, dtype=tf.float32)

    def integrate_causal_reasoning(self, o0):
        """
        Integrates causal reasoning into the Active Inference process by adjusting
        the policy based on the causal graph.
        """
        # Obtain original action probabilities from the habitual network
        Qpi = self.active_inf_model.habitual_net(o0)

        # Adjust the policy based on causal reasoning
        adjusted_Qpi = self.adjust_policy_with_causality(Qpi)

        return adjusted_Qpi

    def update_on_transition(self, o0, pi, o1):
        """
        Updates the causal graph based on the observed transition.
        """
        # Encode initial and resulting observations to obtain state representations
        s0_mean, _ = self.model_down.encoder(o0)
        s1_mean, _ = self.model_down.encoder(o1)
        s0_np = s0_mean.numpy().flatten()
        s1_np = s1_mean.numpy().flatten()
        pi_np = pi.numpy().flatten()

        # Update causal graph with the observed transition
        self.update_causal_graph(s0_np, pi_np, s1_np)

    def save_weights(self, folder_chp):
        """
        Saves the weights of the ActiveInferenceModel and the causal graph.
        """
        self.active_inf_model.save_weights(folder_chp)
        self.save_causal_graph(folder_chp + '/causal_graph.pkl')

    def load_weights(self, folder_chp):
        """
        Loads the weights of the ActiveInferenceModel and the causal graph.
        """
        self.active_inf_model.load_weights(folder_chp)
        self.load_causal_graph(folder_chp + '/causal_graph.pkl')

    def save_causal_graph(self, filepath):
        """
        Saves the causal graph to a file using pickle.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.causal_graph, f)
        self.logger.info(f"Causal graph saved to {filepath}")

    def load_causal_graph(self, filepath):
        """
        Loads the causal graph from a file using pickle.
        """
        with open(filepath, 'rb') as f:
            self.causal_graph = pickle.load(f)
        self.logger.info(f"Causal graph loaded from {filepath}")

    def save_all(self, folder_chp, stats, script_file="", optimizers={}):
        """
        Saves all components including weights, stats, optimizer states, and source files.
        """
        self.save_weights(folder_chp)

        # Save stats
        with open(folder_chp + '/stats.pkl', 'wb') as ff:
            pickle.dump(stats, ff)

        # Save optimizer state using Checkpoint API
        for key, optimizer in optimizers.items():
            checkpoint = tf.train.Checkpoint(optimizer=optimizer)
            checkpoint.save(file_prefix=folder_chp + f'/optimizer_{key}')

        # Copy source files for reference
        copyfile('src/tfmodel.py', folder_chp + '/tfmodel.py')
        copyfile('src/tfloss.py', folder_chp + '/tfloss.py')
        if script_file != "":
            copyfile(script_file, folder_chp + '/' + script_file)

    def load_all(self, folder_chp):
        """
        Loads all components including weights, stats, optimizer states, and source files.
        """
        self.load_weights(folder_chp)

        # Load stats
        with open(folder_chp + '/stats.pkl', 'rb') as ff:
            stats = pickle.load(ff)

        optimizers = {}

        # Restore optimizer state using Checkpoint API
        for key in ['down', 'mid', 'top']:
            optimizer = tf.keras.optimizers.Adam()
            checkpoint = tf.train.Checkpoint(optimizer=optimizer)
            latest_checkpoint = tf.train.latest_checkpoint(folder_chp)
            if latest_checkpoint:
                checkpoint.restore(latest_checkpoint)
            optimizers[key] = optimizer

        # Restore model parameters
        if 'var_beta_s' in stats and len(stats['var_beta_s']) > 0:
            self.model_down.beta_s.assign(stats['var_beta_s'][-1])
        if 'var_gamma' in stats and len(stats['var_gamma']) > 0:
            self.model_down.gamma.assign(stats['var_gamma'][-1])
        if 'var_beta_o' in stats and len(stats['var_beta_o']) > 0:
            self.model_down.beta_o.assign(stats['var_beta_o'][-1])

        return stats, optimizers

    def habitual_net(self, o):
        """
        Computes the habitual network's action probabilities.
        """
        Qpi = self.active_inf_model.habitual_net(o)
        return Qpi

    def imagine_future_from_o(self, o0, pi):
        """
        Imagines the future observation based on the current observation and action.
        """
        return self.active_inf_model.imagine_future_from_o(o0, pi)

    def calculate_G_repeated(self, o, pi, steps=1, calc_mean=False, samples=10):
        """
        Calculates the expected free energy (G) for repeated actions.
        """
        return self.active_inf_model.calculate_G_repeated(o, pi, steps, calc_mean, samples)

    def calculate_G_4_repeated(self, o, steps=1, calc_mean=False, samples=10):
        """
        Calculates the expected free energy (G) for four repeated actions.
        """
        return self.active_inf_model.calculate_G_4_repeated(o, steps, calc_mean, samples)

    def calculate_G(self, s0, pi0, samples=10):
        """
        Calculates the expected free energy (G) for given states and actions.
        """
        return self.active_inf_model.calculate_G(s0, pi0, samples)

    def calculate_G_mean(self, s0, pi0):
        """
        Calculates the expected free energy (G) using mean states.
        """
        return self.active_inf_model.calculate_G_mean(s0, pi0)

    def calculate_G_given_trajectory(self, s0_traj, ps1_traj, ps1_mean_traj, ps1_logvar_traj, pi0_traj):
        """
        Calculates the expected free energy (G) given a trajectory.
        """
        return self.active_inf_model.calculate_G_given_trajectory(s0_traj, ps1_traj, ps1_mean_traj, ps1_logvar_traj, pi0_traj)

    def check_reward(self, o):
        """
        Computes the reward based on the observation.
        """
        if self.active_inf_model.model_down.resolution == 64:
            return tf.reduce_mean(calc_reward(o), axis=[1, 2, 3]) * 10.0
        elif self.active_inf_model.model_down.resolution == 32:
            return tf.reduce_sum(calc_reward_animalai(o), axis=[1, 2, 3])
class ActiveInferenceModel(tf.Module):  # Changed to inherit from tf.Module
    def __init__(self, s_dim, pi_dim, gamma, beta_s, beta_o, colour_channels=1, resolution=64):

        self.tf_precision = tf.float32
        self.precision = 'float32'

        self.s_dim = s_dim
        self.pi_dim = pi_dim
        tf.keras.backend.set_floatx(self.precision)

        if self.pi_dim > 0:
            self.model_top = ModelTop(s_dim, pi_dim, self.tf_precision, self.precision)
            self.model_mid = ModelMid(s_dim, pi_dim, self.tf_precision, self.precision)
        self.model_down = ModelDown(s_dim, pi_dim, self.tf_precision, self.precision, colour_channels, resolution)

        self.model_down.beta_s = tf.Variable(beta_s, trainable=False, name="beta_s")
        self.model_down.gamma = tf.Variable(gamma, trainable=False, name="gamma")
        self.model_down.beta_o = tf.Variable(beta_o, trainable=False, name="beta_o")
        self.pi_one_hot = tf.Variable([[1.0,0.0,0.0,0.0],
                                       [0.0,1.0,0.0,0.0],
                                       [0.0,0.0,1.0,0.0],
                                       [0.0,0.0,0.0,1.0]], trainable=False, dtype=self.tf_precision)
        self.pi_one_hot_3 = tf.Variable([[1.0,0.0,0.0],
                                         [0.0,1.0,0.0],
                                         [0.0,0.0,1.0]], trainable=False, dtype=self.tf_precision)

    # Rest of the class implementation...


    def save_weights(self, folder_chp):
        self.model_down.qs_net.save_weights(folder_chp+'/checkpoint_qs')
        self.model_down.po_net.save_weights(folder_chp+'/checkpoint_po')
        if self.pi_dim > 0:
            self.model_top.qpi_net.save_weights(folder_chp+'/checkpoint_qpi')
            self.model_mid.ps_net.save_weights(folder_chp+'/checkpoint_ps')

    def load_weights(self, folder_chp):
        self.model_down.qs_net.load_weights(folder_chp+'/checkpoint_qs')
        self.model_down.po_net.load_weights(folder_chp+'/checkpoint_po')
        if self.pi_dim > 0:
            self.model_top.qpi_net.load_weights(folder_chp+'/checkpoint_qpi')
            self.model_mid.ps_net.load_weights(folder_chp+'/checkpoint_ps')

    def save_all(self, folder_chp, stats, script_file="", optimizers={}):
        self.save_weights(folder_chp)
        
        # Save stats
        with open(folder_chp + '/stats.pkl', 'wb') as ff:
            pickle.dump(stats, ff)
        
        # Save optimizer state using Checkpoint API
        for key, optimizer in optimizers.items():
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
            checkpoint.save(file_prefix=folder_chp + f'/optimizer_{key}')
        
        # Copy source files for reference
        copyfile('src/tfmodel.py', folder_chp + '/tfmodel.py')
        copyfile('src/tfloss.py', folder_chp + '/tfloss.py')
        if script_file != "":
            copyfile(script_file, folder_chp + '/' + script_file)



    def load_all(self, folder_chp):
        self.load_weights(folder_chp)

        # Load stats
        with open(folder_chp + '/stats.pkl', 'rb') as ff:
            stats = pickle.load(ff)

        optimizers = {}
        
        # Restore optimizer state using Checkpoint API
        try:
            for key in ['down', 'mid', 'top']:  # Replace with actual optimizer names
                optimizer = tf.keras.optimizers.Adam()  # Initialize optimizer
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
                latest_checkpoint = tf.train.latest_checkpoint(folder_chp)
                if latest_checkpoint:
                    checkpoint.restore(latest_checkpoint)
                optimizers[key] = optimizer
        except Exception as e:
            print(f"Error loading optimizers: {e}")
            optimizers = {}

        # Restore model parameters
        if len(stats['var_beta_s']) > 0:
            self.model_down.beta_s.assign(stats['var_beta_s'][-1])
        if len(stats['var_gamma']) > 0:
            self.model_down.gamma.assign(stats['var_gamma'][-1])
        if len(stats['var_beta_o']) > 0:
            self.model_down.beta_o.assign(stats['var_beta_o'][-1])

        return stats, optimizers



    def check_reward(self, o):
        if self.model_down.resolution == 64:
            return tf.reduce_mean(calc_reward(o),axis=[1,2,3]) * 10.0
        elif self.model_down.resolution == 32:
            return tf.reduce_sum(calc_reward_animalai(o), axis=[1,2,3])

    @tf.function
    def imagine_future_from_o(self, o0, pi):
        s0, _, _ = self.model_down.encoder_with_sample(o0)
        ps1, _, _ = self.model_mid.transition_with_sample(pi, s0)
        po1 = self.model_down.decoder(ps1)
        return po1

    @tf.function
    def habitual_net(self, o):
        qs_mean, _ = self.model_down.encoder(o)
        _, Qpi, _ = self.model_top.encode_s(qs_mean)
        return Qpi

    @tf.function
    def calculate_G_repeated(self, o, pi, steps=1, calc_mean=False, samples=10):
        """
        We simultaneously calculate G for the four policies of repeating each
        one of the four actions continuously..
        """
        # Calculate current s_t
        qs0_mean, qs0_logvar = self.model_down.encoder(o)
        qs0 = self.model_down.reparameterize(qs0_mean, qs0_logvar)

        sum_terms = [tf.zeros([o.shape[0]], self.tf_precision), tf.zeros([o.shape[0]], self.tf_precision), tf.zeros([o.shape[0]], self.tf_precision)]
        sum_G = tf.zeros([o.shape[0]], self.tf_precision)

        # Predict s_t+1 for various policies
        if calc_mean: s0_temp = qs0_mean
        else: s0_temp = qs0

        for t in range(steps):
            G, terms, s1, ps1_mean, po1 = self.calculate_G(s0_temp, pi, samples=samples)

            sum_terms[0] += terms[0]
            sum_terms[1] += terms[1]
            sum_terms[2] += terms[2]
            sum_G += G

            if calc_mean:
                s0_temp = ps1_mean
            else:
                s0_temp = s1

        return sum_G, sum_terms, po1

    @tf.function
    def calculate_G_4_repeated(self, o, steps=1, calc_mean=False, samples=10):
        """
        We simultaneously calculate G for the four policies of repeating each
        one of the four actions continuously..
        """
        # Calculate current s_t
        qs0_mean, qs0_logvar = self.model_down.encoder(o)
        qs0 = self.model_down.reparameterize(qs0_mean, qs0_logvar)

        sum_terms = [tf.zeros([4], self.tf_precision), tf.zeros([4], self.tf_precision), tf.zeros([4], self.tf_precision)]
        sum_G = tf.zeros([4], self.tf_precision)

        # Predict s_t+1 for various policies
        if calc_mean: s0_temp = qs0_mean
        else: s0_temp = qs0

        for t in range(steps):
            if calc_mean:
                G, terms, ps1_mean, po1 = self.calculate_G_mean(s0_temp, self.pi_one_hot)
            else:
                G, terms, s1, ps1_mean, po1 = self.calculate_G(s0_temp, self.pi_one_hot, samples=samples)

            sum_terms[0] += terms[0]
            sum_terms[1] += terms[1]
            sum_terms[2] += terms[2]
            sum_G += G

            if calc_mean:
                s0_temp = ps1_mean
            else:
                s0_temp = s1

        return sum_G, sum_terms, po1

    @tf.function
    def calculate_G(self, s0, pi0, samples=10):

        term0 = tf.zeros([s0.shape[0]], self.tf_precision)
        term1 = tf.zeros([s0.shape[0]], self.tf_precision)
        for _ in range(samples):
            ps1, ps1_mean, ps1_logvar = self.model_mid.transition_with_sample(pi0, s0)
            po1 = self.model_down.decoder(ps1)
            qs1, _, qs1_logvar = self.model_down.encoder_with_sample(po1)

            # E [ log P(o|pi) ]
            logpo1 = self.check_reward(po1)
            term0 += logpo1

            # E [ log Q(s|pi) - log Q(s|o,pi) ]
            term1 += - tf.reduce_sum(entropy_normal_from_logvar(ps1_logvar) + entropy_normal_from_logvar(qs1_logvar), axis=1)
        term0 /= float(samples)
        term1 /= float(samples)

        term2_1 = tf.zeros(s0.shape[0], self.tf_precision)
        term2_2 = tf.zeros(s0.shape[0], self.tf_precision)
        for _ in range(samples):
            # Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
            po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0, s0)[0])
            term2_1 += tf.reduce_sum(entropy_bernoulli(po1_temp1),axis=[1,2,3])

            # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
            po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean, ps1_logvar))
            term2_2 += tf.reduce_sum(entropy_bernoulli(po1_temp2),axis=[1,2,3])
        term2_1 /= float(samples)
        term2_2 /= float(samples)

        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2

        G = - term0 + term1 + term2

        return G, [term0, term1, term2], ps1, ps1_mean, po1

    @tf.function
    def calculate_G_mean(self, s0, pi0):

        _, ps1_mean, ps1_logvar = self.model_mid.transition_with_sample(pi0, s0)
        po1 = self.model_down.decoder(ps1_mean)
        _, qs1_mean, qs1_logvar = self.model_down.encoder_with_sample(po1)

        # E [ log P(o|pi) ]
        logpo1 = self.check_reward(po1)
        term0 = logpo1

        # E [ log Q(s|pi) - log Q(s|o,pi) ]
        term1 = - tf.reduce_sum(entropy_normal_from_logvar(ps1_logvar) + entropy_normal_from_logvar(qs1_logvar), axis=1)

        # Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
        po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0, s0)[1])
        term2_1 = tf.reduce_sum(entropy_bernoulli(po1_temp1),axis=[1,2,3])

        # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
        po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean, ps1_logvar))
        term2_2 = tf.reduce_sum(entropy_bernoulli(po1_temp2),axis=[1,2,3])

        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2

        G = - term0 + term1 + term2

        return G, [term0, term1, term2], ps1_mean, po1

    @tf.function
    def calculate_G_given_trajectory(self, s0_traj, ps1_traj, ps1_mean_traj, ps1_logvar_traj, pi0_traj):
        # NOTE: len(s0_traj) = len(s1_traj) = len(pi0_traj)

        po1 = self.model_down.decoder(ps1_traj)
        qs1, _, qs1_logvar = self.model_down.encoder_with_sample(po1)

        # E [ log P(o|pi) ]
        term0 = self.check_reward(po1)

        # E [ log Q(s|pi) - log Q(s|o,pi) ]
        term1 = - tf.reduce_sum(entropy_normal_from_logvar(ps1_logvar_traj) + entropy_normal_from_logvar(qs1_logvar), axis=1)

        #  Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
        po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0_traj, s0_traj)[0])
        term2_1 = tf.reduce_sum(entropy_bernoulli(po1_temp1),axis=[1,2,3])

        # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
        po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean_traj, ps1_logvar_traj))
        term2_2 = tf.reduce_sum(entropy_bernoulli(po1_temp2),axis=[1,2,3])

        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2

        return - term0 + term1 + term2

    #@tf.function
    def mcts_step_simulate(self, starting_s, depth, use_means=False):
        s0 = np.zeros((depth, self.s_dim), self.precision)
        ps1 = np.zeros((depth, self.s_dim), self.precision)
        ps1_mean = np.zeros((depth, self.s_dim), self.precision)
        ps1_logvar = np.zeros((depth, self.s_dim), self.precision)
        pi0 = np.zeros((depth, self.pi_dim), self.precision)

        s0[0] = starting_s
        try:
            Qpi_t_to_return = self.model_top.encode_s(s0[0].reshape(1,-1))[1].numpy()[0]
            pi0[0, np.random.choice(self.pi_dim, p=Qpi_t_to_return)] = 1.0
        except:
            pi0[0, 0] = 1.0
            Qpi_t_to_return = pi0[0]
        ps1_new, ps1_mean_new, ps1_logvar_new = self.model_mid.transition_with_sample(pi0[0].reshape(1,-1), s0[0].reshape(1,-1))
        ps1[0] = ps1_new[0].numpy()
        ps1_mean[0] = ps1_mean_new[0].numpy()
        ps1_logvar[0] = ps1_logvar_new[0].numpy()
        if 1 < depth:
            if use_means:
                s0[1] = ps1_mean_new[0].numpy()
            else:
                s0[1] = ps1_new[0].numpy()
        for t in range(1, depth):
            try:
                pi0[t, np.random.choice(self.pi_dim, p=self.model_top.encode_s(s0[t].reshape(1,-1))[1].numpy()[0])] = 1.0
            except:
                pi0[t, 0] = 1.0
            ps1_new, ps1_mean_new, ps1_logvar_new = self.model_mid.transition_with_sample(pi0[t].reshape(1,-1), s0[t].reshape(1,-1))
            ps1[t] = ps1_new[0].numpy()
            ps1_mean[t] = ps1_mean_new[0].numpy()
            ps1_logvar[t] = ps1_logvar_new[0].numpy()
            if t+1 < depth:
                if use_means:
                    s0[t+1] = ps1_mean_new[0].numpy()
                else:
                    s0[t+1] = ps1_new[0].numpy()

        G = tf.reduce_mean(self.calculate_G_given_trajectory(s0, ps1, ps1_mean, ps1_logvar, pi0)).numpy()
        return G, pi0, Qpi_t_to_return