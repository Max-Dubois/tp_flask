from tkinter import filedialog
import folium
from flask import Flask, render_template, request, send_from_directory
import os
import tkinter as tk
import cv2
import numpy as np
import io
from flask import send_file
from sklearn.cluster import KMeans

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

@app.route('/kmeans', methods=['POST'])
def kmeans():
    image_path = request.form.get('image_path')
    k_clusters = int(request.form.get('k', 5))

    img = cv2.imread(image_path)
    if img is None:
        return "Erreur : Impossible de charger l'image.", 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k_clusters, init="k-means++", n_init=10, max_iter=100, random_state=42)
    kmeans.fit(pixels)

    couleurs_centres = np.uint8(kmeans.cluster_centers_)
    pixels_segmentes = couleurs_centres[kmeans.labels_]
    image_finale = pixels_segmentes.reshape(img_rgb.shape)

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_finale, cv2.COLOR_RGB2BGR))
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=8000)