import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from api.mdb_interact import mdb_interact_bp
from api.init_weight import init_weight_bp
from api.recommend import recommend_bp
from api.train import train_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(init_weight_bp)
app.register_blueprint(recommend_bp)
app.register_blueprint(mdb_interact_bp)
app.register_blueprint(train_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT", 5000), debug=True)