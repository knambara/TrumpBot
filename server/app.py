from flask import Flask
from flask import request

import json

app = Flask(__name__)


@app.route("/trumpbot", methods=["POST"])
def generate_text():
    # semantic = request.args.get('semantic')
    body = request.get_json()['body']
    response = "Make America Great Again"
    return json.dumps({'response': response, "received": body})


if __name__ == "__main__":
    app.run()
