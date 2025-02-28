from flask import Flask, render_template, request, url_for, send_file, jsonify
import os
import tarfile
import zipfile
import tempfile
import shutil
from werkzeug.utils import secure_filename

from src.proxann.proxann import ProxAnn
from src.utils.utils import init_logger

global evaluation_progress
evaluation_progress = 0

app = Flask(__name__)
UPLOAD_FOLDER = "src/metric_mode/uploads"
CONFIG_FOLDER = "src/metric_mode/configs"
CONFIG_PATH="config/config.yaml"
USER_STUDY_CONFIG="config/user_study/config_pilot_test.conf"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONFIG_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logger = init_logger(CONFIG_PATH, f"RunProxann-metric-mode")
proxann = ProxAnn(logger, CONFIG_PATH)

def generate_config(file_paths, trained_with_thetas_eval):
    """ Generates a configuration file dynamically based on uploaded files. """
    if trained_with_thetas_eval:
        config_content = f"""[all]
method=elbow
top_words_display=100
ntop=7
n_matches=4
text_column=tokenized_text
text_column_disp=text
thr=0.1,0.8
path_json_save={UPLOAD_FOLDER}/json_out/tests
topic_selection_method=wmd

[eval]
model_path={file_paths['model']}
corpus_path={file_paths['corpus']}
trained_with_thetas_eval=True
remove_topic_ids=
"""
    else:
        config_content = f"""[all]
method=elbow
top_words_display=100
ntop=7
n_matches=4
text_column=tokenized_text
text_column_disp=text
thr=0.1,0.8
path_json_save={UPLOAD_FOLDER}/json_out/tests
topic_selection_method=wmd

[eval]
thetas_path={file_paths['thetas']}
betas_path={file_paths['betas']}
vocab_path={file_paths['vocab']}
corpus_path={file_paths['corpus']}
trained_with_thetas_eval=False
remove_topic_ids=
"""
    config_path = os.path.join(CONFIG_FOLDER, "config.conf")
    with open(config_path, "w") as conf_file:
        conf_file.write(config_content)
    return config_path

@app.route('/')
def index():
    return render_template('index.html', files=[], progress=0)

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = {}
    trained_with_thetas_eval = 'trained_with_thetas_eval' in request.form
    required_files_proxann = ["model", "corpus"]
    required_files_separate = ["thetas", "betas", "vocab", "corpus"]
    required_files = required_files_proxann if trained_with_thetas_eval else required_files_separate

    valid_file_suffixes = {"npz", "npy", "json", "parquet", "tar.gz", "tar", "zip"}
    
    for file_key in required_files:
        if file_key in request.files:
            file = request.files[file_key]
            filename = secure_filename(file.filename)
            
            if not filename:
                return jsonify({"error": f"Error: No valid filename provided for {file_key}."}), 400

            file_suffix = os.path.splitext(filename)[1].lower().lstrip(".")

            # Enforce file type restrictions
            file_type_map = {
                "thetas": {"npz", "npy"},
                "betas": "npy",
                "vocab": "json",
                "corpus": {"parquet", "json"},
                "model": {"tar.gz", "tar", "zip"}
            }

            expected_type = file_type_map.get(file_key)
            if isinstance(expected_type, set) and file_suffix not in expected_type:
                return jsonify({"error": f"Error: Invalid file type for {file_key}. Expected one of {expected_type}."}), 400
            elif isinstance(expected_type, str) and file_suffix != expected_type:
                return jsonify({"error": f"Error: Invalid file type for {file_key}. Expected '{expected_type}'."}), 400

            # Handle compressed archives
            if file_key == "model":
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)

                try:
                    if file_suffix in {"tar.gz", "tar"}:
                        with tarfile.open(file_path, "r:*") as tar:
                            for member in tar.getmembers():
                                if member.isfile():
                                    member_suffix = os.path.splitext(member.name)[1].lower().lstrip(".")
                                    if member_suffix not in valid_file_suffixes:
                                        raise ValueError(f"Invalid file in archive: {member.name}")
                    elif file_suffix == "zip":
                        with zipfile.ZipFile(file_path, "r") as zip_ref:
                            for member in zip_ref.namelist():
                                member_suffix = os.path.splitext(member)[1].lower().lstrip(".")
                                if member_suffix not in valid_file_suffixes:
                                    raise ValueError(f"Invalid file in archive: {member}")

                    # Move to final upload directory
                    final_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    shutil.move(file_path, final_path)
                    uploaded_files[file_key] = final_path

                except Exception as e:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return jsonify({"error": f"Error: {str(e)}"}), 400

                shutil.rmtree(temp_dir, ignore_errors=True)
                continue

            # Save non-archive files
            final_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(final_path)
            uploaded_files[file_key] = final_path

    if set(required_files) != set(uploaded_files.keys()):
        return jsonify({"error": "Error: Missing required files. Please upload all required files."}), 400

    _ = generate_config(uploaded_files, trained_with_thetas_eval)

    return jsonify({
        "status": "Uploaded Successfully",
        "config_url": url_for('download_config')
    })

@app.route('/download-config')
def download_config():
    config_path = os.path.join(CONFIG_FOLDER, "config.conf")
    return send_file(config_path, as_attachment=True)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """ Evaluation process that generates a DataFrame as results with real-time progress updates. """
    global evaluation_progress
    evaluation_progress = 0
    
    status, tm_model_data_path = proxann.generate_user_provided_json(path_user_study_config_file=USER_STUDY_CONFIG)
    
    if status == 0:
        logger.info("User provided JSON file generated successfully.")
    else:
        logger.error("Error generating user provided JSON file.")
        return jsonify({"error": "Failed to generate user-provided JSON file."}), 500
    
    # Progress Update before Running Metric
    evaluation_progress = 40
    
    df, _ = proxann.run_metric(
        tm_model_data_path.as_posix(),
        llm_models=["qwen:32b"]
    )    
    
    evaluation_progress = 100  # Mark evaluation as complete
    table_html = df.to_html(classes='table table-striped table-bordered text-center', index=False)

    return jsonify({"table": table_html})

@app.route('/evaluate_status', methods=['GET'])
def evaluate_status():
    """ Returns the current progress of the evaluation process. """
    global evaluation_progress
    return jsonify({"progress": evaluation_progress})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
