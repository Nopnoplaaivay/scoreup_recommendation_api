from flask import Blueprint, request, jsonify
from model.mongodb import Database

mdb_interact_bp = Blueprint('mdb_interact', __name__)

db = Database()

@mdb_interact_bp.route("/mdb/update_diff", methods=["POST"])
def update_diff():
    global db
    try:
        db.update_difficulty()
        return jsonify({"status": "success", "message": "Difficulty updated"}), 200
    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing key in request data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@mdb_interact_bp.route("/mdb/update_db", methods=["POST"])
def update_db():
    global db
    try:
        db.encode_exercise_ids()
        db.encode_knowledge_concepts()
        return jsonify({"status": "success", "message": "Database updated"}), 200
    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing key in request data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@mdb_interact_bp.route("/mdb/update_chapter", methods=["POST"])
def update_chapter():
    global db
    try:
        db.reset_logs()
        db.update_chapter()
        return jsonify({"status": "success", "message": "Chapter updated"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@mdb_interact_bp.route("/mdb/db_overview", methods=["GET"])
def db_overview():
    global db
    try:
        num_users = db.users.count_documents({})
        num_questions = db.questions.count_documents({"notionDatabaseId": db.course_id})

        db.update_action_space(cur_chapter="chuong-4")
        num_actions = len(db.action_space)

        return jsonify({
            "status": "success", 
            "num_users": num_users, 
            "num_questions": num_questions, 
            "num_actions": num_actions
        }), 200
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500