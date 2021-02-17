from flask import Flask

app=Flask(__name__)

@app.route("/")
def home():
    return "Hello World!"

if __name__ == '__main__':
    app.run(host='223.194.46.208', port=5000)
