# Importing necessary library
import random

# Class to represent the Prisoners environment
class Prisoners:
    def __init__(self, episode_len, n_round):
        """
        Initialize the Prisoners environment with specified parameters.

        Parameters:
        - episode_len: Length of each episode/game
        - n_round: Number of rounds to play
        """
        self.action_space = 2
        self.n_round = n_round
        self.observation_space = 4
        self.current_round = 0
        self.current_step = 0
        self.episode_len = episode_len
        self.done = False
        self.state = (random.choice([0, 1]), random.choice([0, 1]))
        self.steps = 0
        self.grudge = False

        # Payoff matrix for different actions
        self.payoff_matrix = {(0, 0): (2, 2), (0, 1): (0, 3), (1, 0): (3, 0), (1, 1): (1, 1)}

        # Strategies that the player can follow
        self.strategies = ["Always_cooperate", "Always_defect", "Grudge", "Tit_for_Tat", "Random"]

        self.index = 0

        # Lists to store state and reward information for each round
        self.state_total = []
        self.round_state_list = []
        self.chooosed_startegy_each_round = []
        self.reward_total = []
        self.reward_round_list = []

    def reset(self):
        """
        Reset the environment for a new episode.

        Returns:
        - state: Initial state after reset
        """
        self.state = (random.choice([0, 1]), random.choice([0, 1]))
        if self.index == len(self.strategies):
            self.index = 0
        self.chooosed_startegy_each_round.append(self.strategies[self.index])
        self.current_step = 0
        self.grudge = False
        self.done = False
        self.round_state_list = []
        self.reward_round_list = []
        return self.state

    # Different player strategies
    def coop(self, action):
        return 0

    def defect(self, action):
        return 1

    def Grudge(self, action):
        if action == 1:
            self.grudge = True
        if self.grudge == True:
            return 1
        else:
            return 0

    def Tit_for_Tat(self, action):
        return self.state[0]

    def random_startegy(self):
        return random.choice([0, 1])

    def evaluate(self, action, value):
        """
        Evaluate the environment based on the chosen action and strategy.

        Parameters:
        - action: Action chosen by the player
        - value: Index representing the chosen strategy

        Returns:
        - state: New state after the evaluation
        - reward[0]: Reward for the player
        - done: True if the episode is done, False otherwise
        - reward: Tuple of rewards for both players
        """
        # Print the chosen value for debugging purposes
        print(f"the current chosen value is: {value}")

        player2_action = 0

        # Execute the chosen strategy
        if self.strategies[value] == "Always_cooperate":
            player2_action = self.coop(action)
        elif self.strategies[value] == "Always_defect":
            player2_action = self.defect(action)
        elif self.strategies[value] == "Grudge":
            player2_action = self.Grudge(action)
        elif self.strategies[value] == "Tit_for_Tat":
            player2_action = self.Tit_for_Tat(self.state)
        else:
            player2_action = self.random_startegy()

        # Check if the episode is done
        if self.current_step == self.episode_len:
            print("episode is done")
            self.done = True

        # Update current state, get reward, and increment step
        self.state = (action, player2_action)
        reward = self.payoff_matrix[self.state]
        self.current_step += 1

        return self.state, reward[0], self.done, reward

    def step(self, action):
        """
        Perform a step in the environment based on the chosen action.

        Parameters:
        - action: Action chosen by the player

        Returns:
        - state: New state after the step
        - reward: Reward for the player
        - done: True if the episode is done, False otherwise
        - reward[0]: Reward for the player
        """
        reward = (0, 0)
        player2_action = 0

        # Execute the strategy based on the current index
        if self.strategies[self.index] == "Always_cooperate":
            player2_action = self.coop(action)
        elif self.strategies[self.index] == "Always_defect":
            player2_action = self.defect(action)
        elif self.strategies[self.index] == "Grudge":
            player2_action = self.Grudge(action)
        elif self.strategies[self.index] == "Tit_for_Tat":
            player2_action = self.Tit_for_Tat(action)
        else:
            player2_action = self.random_startegy()

        # Check if the game or episode is over
        if self.current_round == self.n_round:
            print("train over")
        elif self.current_step == self.episode_len:
            self.index += 1
            self.reward_total.append(self.reward_round_list)
            self.state_total.append(self.round_state_list)
            self.current_round += 1
            self.done = True
        else:
            # Update state, increment step, and get reward
            self.state = (action, player2_action)
            self.current_step += 1
            reward = self.payoff_matrix[self.state]
            self.round_state_list.append(self.state)
            self.reward_round_list.append(reward)

        return self.state, reward, self.done, reward[0]
