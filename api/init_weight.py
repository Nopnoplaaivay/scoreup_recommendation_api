from flask import Blueprint, request, jsonify, send_file
from model.init_weight import InitWeight

init_weight_bp = Blueprint('history_train', __name__)

cur_chapter = "chuong-1"
init_weight_agent = InitWeight(cur_chapter="chuong-1")

@init_weight_bp.route("/init_weight", methods=["POST"])
def history_train():
    global cur_chapter, init_weight_agent
    data = request.json
    try:
        cur_chapter_req = data["cur_chapter"]

        '''Update ENV if the chapter changed'''
        if cur_chapter_req != cur_chapter:
            cur_chapter = cur_chapter_req
            init_weight_agent = InitWeight(cur_chapter=cur_chapter)

        '''Train the agent'''
        init_weight_agent.train()
        return jsonify({"status": "success", "message": "Training completed"}), 200

    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing key: {str(e)}"}), 400
    except TypeError as e:
        return jsonify({"status": "error", "message": f"Type error: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"status": "error", "message": f"Value error: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500
    
@init_weight_bp.route("/history_train/score_plot/<cur_chapter>", methods=["GET"])
def score_plot(cur_chapter):
    global init_weight_agent
    try:
        return send_file(f"plots/score_history_{cur_chapter}.png", mimetype="image/png")
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500