from flask import Blueprint, request, jsonify
from model.mongodb import Database
from model.environment import Environment
from model.actor_critic import Agent

recommend_bp = Blueprint('recommend', __name__)

'''Intialize Database and Environment and Agent'''
db = Database()
env = Environment(db, cur_chapter="chuong-1")
agent = Agent(env=env)

@recommend_bp.route("/recommend_action", methods=["POST"])
def recommend_action():
    global env, agent
    data = request.json
    try:
        if "user_id" not in data or "cur_chapter" not in data:
            return jsonify({"status": "error", "message": "Missing key in request data"}), 400
        
        user_id = data["user_id"]
        cur_chapter = data["cur_chapter"]

        '''Update ENV'''
        if cur_chapter != env.cur_chapter:
            env = Environment(db, cur_chapter)
            agent = Agent(env=env)

        '''Convert state'''
        if "state" in data:
            raw_state = data["state"]
            state = env.convert_state(raw_state)
            action = int(agent.choose_action(state))
            exercise = db.questions.find_one({"encoded_exercise_id": action})
            return jsonify({"action": action, "exercise": exercise}), 200
        else:
            chapters_num = cur_chapter.split("-")[-1]
            chapters = [f"chuong-{i}" for i in range(1, int(chapters_num) + 1)]
            user_log_cursor = db.logs.find({"user_id": user_id, "chapter": {"$in": chapters}}).sort("timestamp", -1).limit(1)
            user_logs = list(user_log_cursor)
            if not user_logs:
                raise ValueError("No logs found for the given user and chapters")

            user_log = user_logs[0]
            state = env.extract_state(user_log)[1]
            action = int(agent.choose_action(state))
            exercise = db.questions.find_one({"encoded_exercise_id": action})
            return jsonify({"action": action, "exercise": exercise}), 200
    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing key in request data: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500