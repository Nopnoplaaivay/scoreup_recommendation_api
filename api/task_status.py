from celery.result import AsyncResult
from flask import Blueprint, jsonify


status_bp = Blueprint('status', __name__)

"""Get status of training task"""
@status_bp.route("/status/<task_id>")
def get_status(task_id):
    task = AsyncResult(task_id, app=celery)
    if task.state == "PENDING":
        response = {"state": task.state, "status": "Pending..."}
    elif task.state != "FAILURE":
        response = {"state": task.state, "result": task.result}
    else:
        response = {"state": task.state, "result": str(task.info)}
    return jsonify(response)