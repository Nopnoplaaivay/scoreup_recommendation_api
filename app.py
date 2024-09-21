import tensorflow as tf
import numpy as np
import requests
import os
import json

from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
from collections import defaultdict, deque
from environment import Environment
from actor_critic import Agent
from mongodb import Database
from celery import Celery
from celery.result import AsyncResult
from dotenv import load_dotenv
from print_module import Print
from score_history import Score

load_dotenv()

db = Database()
env = Environment(db)
agent = Agent(env=env)


app = Flask(__name__)
CORS(app)


def make_celery(app):
    celery = Celery(
        app.import_name, backend=os.getenv("REDIS_URL"), broker=os.getenv("REDIS_URL")
    )
    celery.conf.update(app.config)
    return celery


celery = make_celery(app)

"""Online Memory"""
memory = defaultdict(list)

"""API Endpoints"""

score_history, latest_best_score = Score().load_state()
print(f"Best score: {latest_best_score}")

"""Store transition"""
@app.route("/store_transition", methods=["POST"])
def store_transition():
    data = request.json
    user_id = data["user_id"]
    transitions = data["transitions"]

    for transition in transitions:
        """Prepare transition"""
        raw_state = transition["state"][1:]
        raw_next_state = transition["next_state"][1:]
        raw_action = transition["action"]

        state = env.convert_state(raw_state)
        next_state = env.convert_state(raw_next_state)
        action = env.get_action(raw_action)
        done = transition["done"]
        reward = env.reward_func(state, next_state)
        # print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}")

        """Store transition"""
        transition = {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done,
        }

        """Store transition in memory"""
        memory[user_id].append(transition)

    try:
        # Call the task using send_task by specifying the name of the task
        task = celery.send_task('background_train', args=[user_id, json.dumps(memory[user_id])])
        memory[user_id] = []

        return (
            jsonify({
                "status": "success",
                "message": "Transition stored and training started",
                "task_id": task.id,
            }),200,
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

"""Background training task using Celery"""
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
                Print.success(f"Model's new weights saved!")

    """Save score history"""
    Score().save_state(score_history, best_score)

    Print.success(f"Training completed for user {user_id}.")
    return "Training finished"


"""Get status of training task"""
@app.route("/status/<task_id>")
def get_status(task_id):
    task = AsyncResult(task_id, app=celery)
    if task.state == "PENDING":
        response = {"state": task.state, "status": "Pending..."}
    elif task.state != "FAILURE":
        response = {"state": task.state, "result": task.result}
    else:
        response = {"state": task.state, "result": str(task.info)}
    return jsonify(response)


"""Get memory"""
@app.route("/get_memory", methods=["GET"])
def get_memory():
    # Check if memory is empty
    if len(memory) == 0:
        return jsonify({"status": "error", "message": "Memory is empty"}), 404

    length_memory = {
        user_id: len(transitions) for user_id, transitions in memory.items()
    }
    return jsonify(length_memory), 200


@app.route("/get_memory/<user_id>", methods=["GET"])
def get_memory_by_user_id(user_id):
    # Check if user_id exists
    if user_id not in memory:
        return jsonify({"status": "error", "message": "User ID not found"}), 404
    return jsonify(memory[user_id]), 200


"""Predict action"""
@app.route("/predict_action", methods=["POST"])
def predict_action():
    data = request.json
    try:
        raw_state = data["state"]
        state = env.convert_state(raw_state)
        action = int(agent.choose_action(state))
        exercise = db.questions.find_one({"encoded_exercise_id": action})
        return jsonify({"action": action, "exercise": exercise}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

'''Get first action'''
@app.route("/initial_action", methods=["POST"])
def initial_action():
    data = request.json
    try:
        user_id = data["user_id"]
        user_log = db.logs.find({"user_id": user_id}).sort("timestamp", -1).limit(1)[0]
        state = env.extract_state(user_log)[1]
        
        action = int(agent.choose_action(state))
        exercise = db.questions.find_one({"encoded_exercise_id": action})
        return jsonify({"action": action, "exercise": exercise}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
'''Get score_history.png'''
@app.route("/score_history", methods=["GET"])
def score_history():
    try:
        return send_file("plots/score_history.png", mimetype="image/png")
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
'''Run offline_traine.py'''
@app.route("/offline_train", methods=["GET"])
def offline_train():
    try:
        os.system("python offline_train.py")
        return jsonify({"status": "success", "message": "Training completed"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



"""Run Flask app"""
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT", 5000), debug=True)
