import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from model.environment import Environment
from model.mongodb import Database
from model.actor_critic import Agent
from utils.print_module import Print

class InitWeight:
    def __init__(self, cur_chapter):
        self.cur_chapter = cur_chapter
        self.db = Database()
        self.env = Environment(self.db, cur_chapter=cur_chapter)
        self.agent = Agent(env=self.env)
        self.file_name = f"score_history_c{cur_chapter.split('-')[-1]}.png"
        self.figure_file = 'plots/' + self.file_name
        self.load_checkpoint = False
        self.score_history = {}
        self.model_best_score = 0

    def plot_learning_curve(self, score_history):
        try:
            for chapter, users in score_history.items():
                plt.figure()
                for user_id, data in users.items():
                    scores = data["scores"]
                    if not scores:
                        continue

                    x = [i + 1 for i in range(len(scores))]
                    running_avg = np.zeros(len(scores))
                    for i in range(len(running_avg)):
                        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
                    
                    plt.plot(x, running_avg, label=f'User {user_id} Running Avg (100)', linestyle='solid')

                plt.title(f'Learning Curve {chapter}')
                plt.xlabel('No Exercises')
                plt.ylabel('Running Average Score')

                '''Save plot'''
                figure_file = f'plots/score_history_{chapter}.png'
                plt.savefig(figure_file)
                plt.close()
        except Exception as e:
            print(f"An error occurred while plotting the learning curve: {str(e)}")
            raise

    def train(self):
        try:
            users = self.db.users.find()
            users_ids = [str(user['_id']) for user in users]
            '''Initialize model's weights with user's logs''' 
            if os.path.exists("./plots/score_history.json"):
                with open("./plots/score_history.json", "r") as file:
                    self.score_history = json.load(file)

            Print.warning(f"start initializing agent {self.cur_chapter}'s weights...")
            for user_id in users_ids:

                '''Get Chapter Logs of User'''
                chapters_num = self.cur_chapter.split("-")[-1]
                chapters = [f"chuong-{i}" for i in range(1, int(chapters_num) + 1)]
                user_logs = list(self.db.logs.find({"user_id": user_id, "chapter": {"$in": chapters}}).limit(100))
                log_count = len(user_logs)
                if log_count > 0:
                    '''Ensure the score_history dictionary is initialized for the current chapter'''                    
                    if self.cur_chapter not in self.score_history:
                        self.score_history[self.cur_chapter] = {}
                    if user_id not in self.score_history[self.cur_chapter]:
                        self.score_history[self.cur_chapter][user_id] = {"scores": [], "best_score": 0}

                    print(user_id)

                    score_history = []
                    best_score = 0
                    score = 0

                    for idx in range(log_count - 1):
                        done = 1 if idx == log_count - 1 else 0
                        exercise_id, state = self.env.extract_state(user_logs[idx])
                        exercise_id_, next_state = self.env.extract_state(user_logs[idx + 1])
                        reward = self.env.reward_func(state, next_state)
                        
                        '''Get action'''
                        self.agent.action = self.env.get_action(exercise_id_)

                        score += reward
                        if not self.load_checkpoint:
                            self.agent.learn(state, reward, next_state, done)
                        avg_score = np.mean(score_history[-100:]) if len(score_history) > 0 else 0

                        if avg_score > best_score:
                            best_score = avg_score
                            print(f"avg score: {avg_score:.2f}")
                            self.agent.save_models()
                        score_history.append(score)

                        '''Save model's weights'''
                        if best_score > self.model_best_score:
                            self.model_best_score = best_score

                    self.score_history[self.cur_chapter][user_id] = {"scores": score_history, "best_score": best_score}
                    # Check keys
                    print(f"user_id: {user_id} - {log_count} - best_score: {best_score:.2f}")
        

                '''Save score history to json file'''
                with open("./plots/score_history.json", "w") as file:
                    json.dump(self.score_history, file, indent=4)

            '''Plot combined learning curve for all chapters and users'''
            self.plot_learning_curve(self.score_history)
            Print.success(f"{self.cur_chapter}'s weights initialized successfully")
        
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500