from flask import Flask
from routes.yolo_routes import yolo_bp

app = Flask(__name__)

# Register the blueprint
app.register_blueprint(yolo_bp)

if __name__ == '__main__':
    app.run(debug=True, port=5000)