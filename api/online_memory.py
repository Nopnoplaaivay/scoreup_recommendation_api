import json
from flask import Blueprint, request, jsonify
from collections import defaultdict
from model.environment import Environment
from model.actor_critic import Agent
from model.mongodb import Database
from state.score_history import Score

memory_bp = Blueprint('memory', __name__)

memory = defaultdict(list)

# db = Database()
# env = Environment(db, cur_chapter="chuong-1")
# agent = Agent(env=env)

# def init_celery(celery):
#     global celery_instance
#     celery_instance = celery

# """Store transition"""
# @memory_bp.route("/store_transition", methods=["POST"])
# def store_transition():
#     data = request.json
#     user_id = data["user_id"]
#     transitions = data["transitions"]

#     for transition in transitions:
#         """Prepare transition"""
#         raw_state = transition["state"][1:]
#         raw_next_state = transition["next_state"][1:]
#         raw_action = transition["action"]

#         state = env.convert_state(raw_state)
#         next_state = env.convert_state(raw_next_state)
#         action = env.get_action(raw_action)
#         done = transition["done"]
#         reward = env.reward_func(state, next_state)
#         # print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}")

#         """Store transition"""
#         transition = {
#             "state": state,
#             "action": action,
#             "next_state": next_state,
#             "reward": reward,
#             "done": done,
#         }

#         """Store transition in memory"""
#         memory[user_id].append(transition)

#     try:
#         # Call the task using send_task by specifying the name of the task
#         task = celery_instance.send_task('background_train', args=[user_id, json.dumps(memory[user_id])])
#         memory[user_id] = []

#         return (
#             jsonify({
#                 "status": "success",
#                 "message": "Transition stored and training started",
#                 "task_id": task.id,
#             }),200,
#         )

#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500


# """Get memory"""
# @memory_bp.route("/get_memory", methods=["GET"])
# def get_memory():
#     # Check if memory is empty
#     if len(memory) == 0:
#         return jsonify({"status": "error", "message": "Memory is empty"}), 404

#     length_memory = {
#         user_id: len(transitions) for user_id, transitions in memory.items()
#     }
#     return jsonify(length_memory), 200

# @memory_bp.route("/get_memory/<user_id>", methods=["GET"])
# def get_memory_by_user_id(user_id):
#     # Check if user_id exists
#     if user_id not in memory:
#         return jsonify({"status": "error", "message": "User ID not found"}), 404
#     return jsonify(memory[user_id]), 200

# """Background training task using Celery"""
# @celery_instance.task(name="background_train")
# def background_train(user_id, transitions_json):
#     print(f"Training model for user {user_id}...")
#     transitions = json.loads(transitions_json)

#     score = 0
#     load_checkpoint = False
#     best_score = latest_best_score

#     for transition in transitions:
#         state = transition["state"]
#         action = transition["action"]
#         next_state = transition["next_state"]
#         reward = transition["reward"]
#         done = transition["done"]

#         """Set agent action"""
#         agent.action = action

#         """Learn"""
#         score += reward
#         if not load_checkpoint:
#             agent.learn(state, reward, next_state, done)
#         score_history.append(score)
#         avg_score = np.mean(score_history[-100:])

#         if avg_score > best_score:
#             best_score = avg_score
#             if not load_checkpoint:
#                 agent.save_models()
#                 Print.success(f"Model's new weights saved!")

#     """Save score history"""
#     Score().save_state(score_history, best_score)

#     Print.success(f"Training completed for user {user_id}.")
#     return "Training finished"
