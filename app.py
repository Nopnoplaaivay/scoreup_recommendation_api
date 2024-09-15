import tensorflow as tf
import numpy as np
import requests
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict, deque
from environment import Environment
from actor_critic import Agent
from mongodb import Database
from celery import Celery
from celery.result import AsyncResult
from print_module import Print

db = Database()
env = Environment(db)
agent = Agent(env=env)

app = Flask(__name__)
CORS(app)

# def make_celery(app):
#     celery = Celery(
#         app.import_name,
#         backend='redis://127.0.0.1:6379/0',
#         broker='redis://127.0.0.1:6379/0'
#     )
#     celery.conf.update(app.config)
#     return celery

# celery = make_celery(app)

'''Online Memory'''
memory = defaultdict(list)

'''API Endpoints'''

score_history = []


'''Store transition'''
@app.route("/store_transition", methods=["POST"])
def store_transition():
    data = request.json
    user_id = data["user_id"]
    transitions = data["transitions"]
    
    for transition in transitions:

        '''Prepare transition'''
        raw_state = transition["state"][1:]
        raw_next_state = transition["next_state"][1:]
        raw_action = transition["action"]
        
        state = env.convert_state(raw_state)
        next_state = env.convert_state(raw_next_state)
        action = env.get_action(raw_action)
        done = transition["done"]
        reward = env.reward_func(state, next_state)
        # print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}")

        '''Store transition'''
        transition = {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done
        }

        '''Store transition in memory'''
        memory[user_id].append(transition)

    # try:
    #     res = requests.post("http://127.0.0.1:8019/train", json={"user_id": user_id})
    #     if res.status_code != 200:
    #         return jsonify({"status": "error", "message": "Failed to start training"}), 500
    #     return jsonify({"status": "success", "message": "Transition stored and training started"}), 200

    # except Exception as e:
    #     return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "success"}), 200

# Train API that uses Celery to train in the background
# @app.route("/train", methods=["POST"])
# def train_model():
#     user_id = request.json["user_id"]

#     if user_id not in memory:
#         return jsonify({"message": "No transitions to train"}), 400
    
#     print(f"Transitions for user {user_id}:")
#     for transition in memory[user_id]:
#         print(transition)

#     # Pass the transitions to the background training task
#     task = background_train.apply_async(args=[user_id, memory[user_id]])
    
#     return jsonify({"message": "Training started!", "task_id": task.id})

# '''Background training task using Celery'''
# @celery.task
# def background_train(user_id, transitions):
#     print(f"Training model for user {user_id}...")

#     score = 0
#     load_checkpoint = False

#     for transition in transitions:
#         state = transition["state"]
#         action = transition["action"]
#         next_state = transition["next_state"]
#         reward = transition["reward"]
#         done = transition["done"]
        
#         '''Set agent action'''
#         agent.action = action

#         '''Learn'''
#         score += reward
#         if not load_checkpoint:
#             agent.learn(state, reward, next_state, done)
#         score_history.append(score)
#         avg_score = np.mean(score_history[-100:])

#         if avg_score > best_score:
#             best_score = avg_score
#             if not load_checkpoint:
#                 agent.save_models()
        
#     Print.success(f"Training completed for user {user_id}.")
#     return "Training finished"

# '''Get status of training task'''
# @app.route("/status/<task_id>")
# def get_status(task_id):
#     task = AsyncResult(task_id, app=celery)
#     if task.state == "PENDING":
#         response = {"state": task.state, "status": "Pending..."}
#     elif task.state != "FAILURE":
#         response = {"state": task.state, "result": task.result}
#     else:
#         response = {"state": task.state, "result": str(task.info)}
#     return jsonify(response)

'''Get memory'''
@app.route("/get_memory", methods=["GET"])
def get_memory():
    # Check if memory is empty
    if len(memory) == 0:
        return jsonify({"status": "error", "message": "Memory is empty"}), 404
    
    length_memory = {user_id: len(transitions) for user_id, transitions in memory.items()}
    return jsonify(length_memory), 200

@app.route("/get_memory/<user_id>", methods=["GET"])
def get_memory_by_user_id(user_id):
    # Check if user_id exists 
    if user_id not in memory:
        return jsonify({"status": "error", "message": "User ID not found"}), 404
    return jsonify(memory[user_id]), 200


'''Predict action'''
@app.route('/predict_action', methods=['POST'])
def predict_action():
    data = request.json
    try:
        raw_state = data['state']   
        state = env.convert_state(raw_state)
        action = int(agent.choose_action(state))
        exercise = db.questions.find_one({"encoded_exercise_id": action})
        return jsonify({"action": action, "exercise": exercise}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

'''Run Flask app'''
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv('PORT', 5000), debug=True)