from tkinter import filedialog
import folium
from flask import Flask, render_template, request, send_from_directory, send_file
import os
import tkinter as tk
import cv2
import numpy as np
import io
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
    k = request.form.get('k', 5)
    return render_template('resultat.html', image_path=image_path, k=k)

@app.route('/result_kmeans')
def generate_kmeans_image():
    image_path = request.args.get('image_path')
    k_clusters = int(request.args.get('k', 5))

    img = cv2.imread(image_path)
    if img is None: return "Erreur", 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    yy, xx = np.meshgrid(np.linspace(0, 255, h), np.linspace(0, 255, w), indexing='ij')
    
    data_full = np.dstack((img_rgb, xx, yy)).reshape((-1, 5))

    scale_percent = 30 
    img_small = cv2.resize(img_rgb, (int(w*scale_percent/100), int(h*scale_percent/100)), interpolation=cv2.INTER_AREA)
    sh, sw, _ = img_small.shape
    s_yy, s_xx = np.meshgrid(np.linspace(0, 255, sh), np.linspace(0, 255, sw), indexing='ij')
    data_small = np.dstack((img_small, s_xx, s_yy)).reshape((-1, 5))

    kmeans = KMeans(n_clusters=k_clusters, n_init=1, max_iter=10, random_state=42)
    kmeans.fit(data_small)
    
    labels_full = kmeans.predict(data_full)

    couleurs_centres = np.uint8(kmeans.cluster_centers_[:, :3])
    image_finale = couleurs_centres[labels_full].reshape(img_rgb.shape)

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_finale, cv2.COLOR_RGB2BGR))
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=8000)