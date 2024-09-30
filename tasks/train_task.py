import time
import numpy as np

from utils.print_module import Print

def train_model(agent, req, batch):
    Print.warning("Training model...")
    cur_chapter = req["chapter"]
    user_id = req["user_id"]
    print(f"Current chapter: {cur_chapter}")
    print(f"User ID: {user_id}")
    
    load_checkpoint = False
    best_score = 0
    score_history = []
    score = 0

    for transition in batch:
        state = transition["state"]
        action = transition["action"]
        next_state = transition["next_state"]
        reward = transition["reward"]
        done = transition["done"]

        agent.action = action

        score += reward
        if not load_checkpoint:
            agent.learn(state, reward, next_state, done)
        score_history.append(score)
        avg_score = np.mean(score_history[-10:])

        if avg_score > best_score:
            best_score = avg_score
            print(f"{avg_score:.2f}")
            if not load_checkpoint:
                agent.save_models()
    Print.success("Training complete!")

