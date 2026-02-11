import folium
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('accueil.html')

@app.route('/carte',methods=['GET', 'POST'])
def about():
    m = folium.Map(location=[50.952, 1.881], zoom_start=15)
    folium.Marker([50.952, 1.881], popup='Je suis ici !').add_to(m)
    m.save('static/map.html')
    return render_template('carte.html')

# @app.route('/about')
# def about():
#     return "À propos de moi : Je suis en train d'apprendre à développer des applications web avec Flask !"

if __name__ == '__main__':
    app.run(debug=True, port=8000)