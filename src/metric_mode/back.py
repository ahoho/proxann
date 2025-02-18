from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html', files=[], progress=0)

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return redirect(request.url)
    uploaded_files = request.files.getlist('files')
    file_names = []
    progress = 0
    total_files = len(uploaded_files)
    
    for index, file in enumerate(uploaded_files):
        if file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            file_names.append(file.filename)
            progress = int(((index + 1) / total_files) * 100)
    
    return render_template('index.html', files=file_names, status="Uploaded Successfully", progress=progress)
    
if __name__ == '__main__':
    app.run(debug=True)