from tkinter import filedialog
import folium
from flask import Flask, render_template, request, send_from_directory
import os
import tkinter as tk

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('accueil.html')

@app.route('/carte')
def carte():
    m = folium.Map(location=[50.952, 1.881], zoom_start=15)
    folium.Marker([50.952, 1.881], popup='Je suis ici !').add_to(m)
    m.save('static/map.html')
    return render_template('carte.html')

@app.route('/galerie', methods=['GET', 'POST'])
def galerie():
    images = []
    folder_path = ""
    
    if request.method == 'POST':
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory()
        root.destroy()

        if folder_path:
            valid_exts = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(valid_exts):
                    images.append(filename)
                
    return render_template('galerie.html', images=images, folder_path=folder_path)

@app.route('/image/<path:full_path>')
def serve_image(full_path):
    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    app.run(debug=True, port=8000)