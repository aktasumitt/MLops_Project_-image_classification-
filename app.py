from flask import Flask, render_template, request, redirect, send_from_directory 
import os
from werkzeug.utils import secure_filename
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.test_pipeline import TestPipeline
from src.config.configuration import Configuration

# Klasör ayarları
configuration = Configuration()
UPLOAD_FOLDER = configuration.get_prediction_configs().local_data_folder

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def homePage():
    results = None
    test_results = None
    upload_message = ""

    if request.method == 'POST':
        action = request.form.get("action")

        if action == "upload":
            if 'files' not in request.files:
                return redirect(request.url)
            files = request.files.getlist('files')  # Birden fazla dosya al
            for file in files:
                if file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            upload_message = "Images uploaded successfully!"

        elif action == "train":
            training_pipeline = TrainingPipeline()
            training_pipeline.run_training_pipeline()
            return render_template("index.html", images=os.listdir(UPLOAD_FOLDER), message="Training Successful!")

        elif action == "predict":
            prediction_pipeline = PredictionPipeline()
            results = prediction_pipeline.run_prediction_pipeline()
            return render_template("index.html", images=os.listdir(UPLOAD_FOLDER), results=results, prediction_success=True)

        elif action == "test":
            test_pipeline = TestPipeline()
            test_results = test_pipeline.run_test()
            return render_template("index.html", images=os.listdir(UPLOAD_FOLDER), test_results=test_results, test_success=True)

        elif action == "delete":
            files_to_delete = request.form.getlist("delete_files")  # Seçilen dosyaları al
            for file in files_to_delete:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                if os.path.exists(file_path):
                    os.remove(file_path)

    images = sorted(os.listdir(UPLOAD_FOLDER))
    return render_template("index.html", images=images, upload_message=upload_message)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
