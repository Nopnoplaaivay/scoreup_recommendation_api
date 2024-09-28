import json
import numpy as np
import os
from celery import Celery
from model.actor_critic import Agent
from model.environment import Environment
from model.mongodb import Database
from state.score_history import Score
from ..utils.print_module import Print

# Initialize Celery
celery = Celery(
    __name__, backend=os.getenv("REDIS_URL"), broker=os.getenv("REDIS_URL")
)

# Initialize Database, Environment, and Agent
db = Database(cur_chapter='chuong-1')
env = Environment(db)
agent = Agent(env=env)

# Load score history and best score
score_history, latest_best_score = Score().load_state()
print(f"Best score: {latest_best_score}")

@celery.task(name="background_train")
def background_train(user_id, transitions_json):
    print(f"Training model for user {user_id}...")
    transitions = json.loads(transitions_json)

    score = 0
    load_checkpoint = False
    best_score = latest_best_score

    for transition in transitions:
        state = transition["state"]
        action = transition["action"]
        next_state = transition["next_state"]
        reward = transition["reward"]
        done = transition["done"]

        """Set agent action"""
        agent.action = action

        """Learn"""
        score += reward
        if not load_checkpoint:
            agent.learn(state, reward, next_state, done)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()