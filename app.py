import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# from api.online_memory import memory_bp
from api.mdb_interact import mdb_interact_bp
from api.init_weight import init_weight_bp
from api.recommend import recommend_bp
# from api.task_status import status_bp
# from tasks.celery_config import make_celery



app = Flask(__name__)
CORS(app)

# celery = make_celery(app)

# Register Blueprints
# app.register_blueprint(memory_bp)
app.register_blueprint(init_weight_bp)
app.register_blueprint(recommend_bp)
app.register_blueprint(mdb_interact_bp)
# app.register_blueprint(status_bp)
# app.register_blueprint(offline_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT", 5000), debug=True)