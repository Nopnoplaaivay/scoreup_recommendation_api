import numpy as np
import matplotlib.pyplot as plt

from environment import Environment
from mongodb import Database
from actor_critic import Agent
from print_module import Print

def plot_learning_curve(x, scores, figure_file):
    # Calculate the running average of the last 100 scores
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    
    # Create the plot
    plt.plot(x, scores, label='Score')
    plt.plot(x, running_avg, label='Running Average (100)', color='orange')
    plt.title('Learning Curve')
    plt.xlabel('User ID')
    plt.ylabel('Score')
    plt.legend(loc='upper left')

    # Save the plot to the specified file
    plt.savefig(figure_file)
    plt.show()

if __name__ == '__main__':

    '''
    Set up DB and ENV
    '''
    db = Database()
    env = Environment(db)
    agent = Agent(env=env)

    file_name = 'actor_critic.png'
    figure_file = 'plots/' + file_name

    load_checkpoint = False
    best_score = 0
    score_history = []


    users = db.users.find()
    users_ids = [str(user['_id']) for user in users]
    # print(agent.action_space)

    '''Train the agent'''    
    Print.success("Training the agent")
    for user_id in users_ids:
        user_logs = list(db.logs.find({"user_id": user_id}).limit(100))
        # log_count = user_logs.count_documents({"user_id": user_id})
        log_count = len(user_logs)
        if log_count > 0:
            print("--------------------")
            print(f"User ID: {user_id}")
            print(f"User has answered {log_count} questions")

            score = 0
            for idx in range(log_count - 1):
                done = 1 if idx == log_count - 1 else 0
                # try:                
                exercise_id, state = env.extract_state(user_logs[idx])
                exercise_id_, next_state = env.extract_state(user_logs[idx + 1])
                reward = env.reward_func(state, next_state)
                
                '''Get action'''
                agent.action = env.get_action(exercise_id_)

                score += reward
                if not load_checkpoint:
                    agent.learn(state, reward, next_state, done)
                score_history.append(score)
                avg_score = np.mean(score_history[-100:])

                if avg_score > best_score:
                    best_score = avg_score
                    if not load_checkpoint:
                        agent.save_models()

                print(f"Reward: {reward} | Score: {score} | Best Score: {best_score} | Avg Score: {avg_score}")
    Print.success("Training completed")

    '''Plot the learning curve'''
    if not load_checkpoint:
        x = [i+1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)
        Print.success(f"Plot saved to {figure_file}")
        
    