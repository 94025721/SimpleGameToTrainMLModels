import json
import os.path

import matplotlib.pyplot as plt
import datetime


class ScoreTracker:
    def __init__(self):
        """
        Initialize the ScoreTracker class and create a new score file.
        """
        self.create_new_score_file()
        self.initialize_score_file()

    def create_new_score_file(self):
        """
        Create a new score file with the current date and time in the filename.
        """
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.score_file = os.path.join("plotter", f"trainings_data_{current_time}.json")
        os.makedirs(os.path.dirname(self.score_file), exist_ok=True)

    def initialize_score_file(self):
        """
        Initialize the score file with an empty list of scores.
        """
        scores = []
        with open(self.score_file, 'w') as f:
            json.dump(scores, f)

    def save_episode_data(self, episode, score, epsilon, finish_count):
        """
        Save episode data to the score file.

        Args:
            episode (int): The episode number from the training session.
            score (float): The score achieved in the episode.
            epsilon (float): The epsilon value used in the episode.
            finish_count (int): The number of times the task was finished in the episode.
        """
        with open(self.score_file, 'r') as f:
            data = json.load(f)

        data.append({
            'Episode': episode,
            'Score': score,
            'Epsilon': epsilon,
            'FinishCount': finish_count
        })

        with open(self.score_file, 'w') as f:
            json.dump(data, f)

    def plot_scores_and_epsilon(self):
        """
        Plot the scores and epsilon values over episodes.

        The plot will show the score and epsilon on two different y-axes.
        """
        try:
            with open(self.score_file, 'r') as f:
                data = json.load(f)

            episodes = [entry['Episode'] for entry in data]
            scores = [entry['Score'] for entry in data]
            epsilons = [entry['Epsilon'] for entry in data]
            finish_counts = [entry['FinishCount'] for entry in data]

            fig, ax1 = plt.subplots()

            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Score', color='tab:blue')
            ax1.plot(episodes, scores, color='tab:blue', label='Score')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Epsilon', color='tab:orange')
            ax2.plot(episodes, epsilons, color='tab:orange', label='Epsilon')
            ax2.tick_params(axis='y', labelcolor='tab:orange')

            ax3 = ax1.twinx()
            ax3.set_ylabel('Finish Count', color='tab:green')
            ax3.plot(episodes, finish_counts, color='tab:green', label='Finish Count')
            ax3.tick_params(axis='y', labelcolor='tab:green')

            fig.tight_layout()
            plt.title('Training Scores, Epsilon, and Finish Count Over Time')
            plt.show()
        except FileNotFoundError:
            print(f"No score file found at {self.score_file}.")
