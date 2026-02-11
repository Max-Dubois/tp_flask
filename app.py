from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('accueil.html')

@app.route('/carte')
def about():
    return render_template('carte.html')

# @app.route('/about')
# def about():
#     return "À propos de moi : Je suis en train d'apprendre à développer des applications web avec Flask !"

if __name__ == '__main__':
    app.run(debug=True, port=8000)