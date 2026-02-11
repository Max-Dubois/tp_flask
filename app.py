import folium
from flask import Flask, render_template, request

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
    if request.method == 'POST':
        files = request.files.getlist("folder_files")
        
        for file in files:
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                import base64
                file_content = file.read()
                base64_pic = base64.b64encode(file_content).decode('utf-8')
                images.append(f"data:{file.content_type};base64,{base64_pic}")
                
    return render_template('galerie.html', images=images)

if __name__ == '__main__':
    app.run(debug=True, port=8000)