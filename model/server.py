import json

from flask import Flask, jsonify, request
from interact import Bot

app = Flask(__name__)

bot = Bot()

@app.route('/trumpbot', methods=['POST'])
def generate_text():
    content = request.json

    if 'prompt' not in content:
        return 'No prompt provided', 400

    prompt = content['prompt']
    if len(prompt.split()) >= 50:
        return 'Prompt is too long. Try a prompt less than 50 words', 400

    answer = bot.answer(prompt)

    return jsonify(answer=answer, prompt=prompt)


if __name__ == '__main__':
    app.run()