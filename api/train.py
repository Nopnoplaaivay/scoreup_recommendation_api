from flask import Blueprint, request, jsonify
from rq import Queue
from redis import Redis
from tasks.train_task import train_model
from model.mongodb import Database
from model.environment import Environment
from model.actor_critic import Agent
from memory.online_memory import OnlineMemory

'''Intialize Database and Environment and Agent'''
db = Database()
env = Environment(db, cur_chapter="chuong-1")
agent = Agent(env=env)
agent.load_models()
onl_memory = OnlineMemory(env=env)

redis_conn = Redis(host='101.96.66.218', port=8002, db=0, password='21012004Viet')
task_queue = Queue("task_queue", connection=redis_conn)

train_bp = Blueprint('train', __name__)

@train_bp.route('/train', methods=['POST'])
def train():
    global env, agent, onl_memory
    req = request.json
    cur_chapter = req["chapter"]
    user_id = req["user_id"]
    transitions = req["transitions"]

    '''Update ENV'''
    if cur_chapter != env.cur_chapter:
        env = Environment(db, cur_chapter=cur_chapter)
        agent = Agent(env=env)
        agent.load_models()

    '''Add transition to Memory'''
    onl_memory.process_transitions(req)
    batch = onl_memory.batch
    onl_memory.reset()

    try:
        job = task_queue.enqueue(train_model, agent, req, batch)
        print(f"Enqueued task with job ID: {job.id}")
        return jsonify({"message": "Task enqueued", "job_id": job.id}), 202
    except Exception as e:
        return f"Error: {str(e)}"