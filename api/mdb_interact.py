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