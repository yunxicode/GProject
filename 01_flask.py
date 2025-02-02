from flask import Flask, render_template, send_from_directory, request, url_for, flash, redirect
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'C:/Users/yulin/anaconda3/GProject_/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf'])

@app.route("/")
def index():
    return render_template("index.html")
@app.route('/Background.png')
def background(): 
    return send_from_directory(os.path.join(app.root_path, '/static'), 'Background.png')
@app.route('/favicon.svg')
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, '/static'), 'favicon.svg')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/file-upload', methods=['GET', 'POST'])

def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',filename=filename))
            
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

if __name__ == '__main__':
    app.run()