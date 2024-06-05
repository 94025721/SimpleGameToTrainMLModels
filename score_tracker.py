import json
import matplotlib.pyplot as plt


class ScoreTracker:
    def __init__(self, score_file='scores.json'):
        self.score_file = score_file

    def initialize_score_file(self):
        scores = []
        with open(self.score_file, 'w') as f:
            json.dump(scores, f)

    def save_episode_data(self, episode, score, epsilon):
        with open(self.score_file, 'r') as f:
            data = json.load(f)

        data.append({'Episode': episode, 'Score': score, 'Epsilon': epsilon})

        with open(self.score_file, 'w') as f:
            json.dump(data, f)

    def plot_scores_and_epsilon(self):
        try:
            with open(self.score_file, 'r') as f:
                data = json.load(f)

            episodes = [entry['Episode'] for entry in data]
            scores = [entry['Score'] for entry in data]
            epsilons = [entry['Epsilon'] for entry in data]

            fig, ax1 = plt.subplots()

            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Score', color='tab:blue')
            ax1.plot(episodes, scores, color='tab:blue', label='Score')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Epsilon', color='tab:orange')
            ax2.plot(episodes, epsilons, color='tab:orange', label='Epsilon')
            ax2.tick_params(axis='y', labelcolor='tab:orange')

            fig.tight_layout()
            plt.title('Training Scores and Epsilon Over Time')
            plt.show()
        except FileNotFoundError:
            print(f"No score file found at {self.score_file}.")
