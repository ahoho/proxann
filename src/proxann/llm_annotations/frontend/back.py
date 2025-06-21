from datetime import datetime, timedelta
import time
from flask import Flask, url_for, render_template, request, send_file, jsonify # type: ignore
import os
import tarfile
import zipfile
import tempfile
import shutil
import re
from pathlib import Path
from werkzeug.utils import secure_filename # type: ignore
import threading
import uuid
import pandas as pd

from proxann.llm_annotations.utils import is_openai_key_valid
from proxann.llm_annotations.proxann import ProxAnn
from proxann.utils.file_utils import init_logger

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CONFIG_PATH = "src/proxann/config/config.yaml"
logger = init_logger(CONFIG_PATH, "RunProxann-metric-mode")
proxann = ProxAnn(logger, CONFIG_PATH)

task_results = {}

def cleanup_old_tasks(expiration_minutes=30):
    while True:
        now = datetime.utcnow()
        to_delete = []
        for task_id, data in task_results.items():
            task_time = data.get("timestamp")
            if task_time and now - task_time > timedelta(minutes=expiration_minutes):
                to_delete.append(task_id)
        for task_id in to_delete:
            del task_results[task_id]
            shutil.rmtree(os.path.join(UPLOAD_FOLDER, task_id), ignore_errors=True)
        time.sleep(300)

def format_column_label_html(col_name):
    if col_name == "id":
        return "<i>k</i>"

    pattern = r"(rank|fit)_(tau|ndcg|agree)_tm_(.+)"
    match = re.match(pattern, col_name)

    if not match:
        return col_name  # fallback if not matching

    mode, metric, _ = match.groups()

    metric_symbols = {
        "tau": "&tau;",
        "ndcg": "NDCG",
        "agree": "Agree"
    }

    prefix = f'<span style="font-variant: small-caps;">{mode}-</span>'
    metric = metric_symbols[metric]
    subscript = f"<sub>lm:tm</sub>"
    k_rendered = "(<i>k</i>)"

    return f"{prefix}{metric}{subscript}{k_rendered}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    os.makedirs(task_dir, exist_ok=True)

    trained_with_thetas_eval = 'trained_with_thetas_eval' in request.form
    required_files = ["model", "corpus"] if trained_with_thetas_eval else ["thetas", "betas", "vocab", "corpus"]
    uploaded_files = {}
    valid_file_suffixes = {"npz", "npy", "json", "parquet", "tar.gz", "tar", "zip"}

    for file_key in required_files:
        if file_key in request.files:
            file = request.files[file_key]
            filename = secure_filename(file.filename)
            if not filename:
                return jsonify({"error": f"No filename for {file_key}"}), 400
            file_suffix = os.path.splitext(filename)[1].lower().lstrip(".")
            file_type_map = {
                "thetas": {"npz", "npy"},
                "betas": "npy",
                "vocab": "json",
                "corpus": {"parquet", "json"},
                "model": {"tar.gz", "tar", "zip"}
            }
            expected = file_type_map[file_key]
            if isinstance(expected, set) and file_suffix not in expected or isinstance(expected, str) and file_suffix != expected:
                return jsonify({"error": f"Invalid file type for {file_key}."}), 400

            final_path = os.path.join(task_dir, filename)

            if file_key == "model":
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, filename)
                file.save(temp_path)
                try:
                    if file_suffix in {"tar.gz", "tar"}:
                        with tarfile.open(temp_path, "r:*") as tar:
                            for m in tar.getmembers():
                                if m.isfile():
                                    ext = os.path.splitext(m.name)[1].lower().lstrip(".")
                                    if ext not in valid_file_suffixes:
                                        raise ValueError(f"Invalid file in archive: {m.name}")
                    elif file_suffix == "zip":
                        with zipfile.ZipFile(temp_path, "r") as zip_ref:
                            for m in zip_ref.namelist():
                                ext = os.path.splitext(m)[1].lower().lstrip(".")
                                if ext not in valid_file_suffixes:
                                    raise ValueError(f"Invalid file in archive: {m}")
                    shutil.move(temp_path, final_path)
                    uploaded_files[file_key] = final_path
                except Exception as e:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return jsonify({"error": str(e)}), 400
                shutil.rmtree(temp_dir)
            else:
                file.save(final_path)
                uploaded_files[file_key] = final_path

    if set(required_files) != set(uploaded_files.keys()):
        return jsonify({"error": "Missing required files."}), 400

    # Save config file
    text_column_disp = request.form.get("text_column_disp", "text").strip()

    config_content = f"""[all]
method=elbow
top_words_display=100
ntop=7
n_matches=-1
text_column=tokenized_text
text_column_disp={text_column_disp}
thr=0.1,0.8
path_json_save={task_dir}
topic_selection_method=wmd

[eval]
{"model_path" if trained_with_thetas_eval else "thetas_path"}={uploaded_files['model' if trained_with_thetas_eval else 'thetas']}
{"corpus_path" if trained_with_thetas_eval else "betas_path"}={uploaded_files['corpus' if trained_with_thetas_eval else 'betas']}
{"vocab_path=" + uploaded_files["vocab"] if not trained_with_thetas_eval else ""}
corpus_path={uploaded_files['corpus']}
trained_with_thetas_eval={trained_with_thetas_eval}
remove_topic_ids=
"""
    config_path = os.path.join(task_dir, "config.conf")
    with open(config_path, "w") as f:
        f.write(config_content)

    return jsonify({
        "status": "Uploaded",
        "task_id": task_id,
        "config_url": url_for('download_config', task_id=task_id)
    })


@app.route('/download-config/<task_id>')
def download_config(task_id):
    config_path = os.path.join(app.config['UPLOAD_FOLDER'], task_id, "config.conf")
    return send_file(config_path, as_attachment=True)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    task_id = request.form.get("task_id")
    task_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    if not os.path.exists(task_dir):
        return jsonify({"error": "Invalid task ID"}), 400

    llm_model = request.form.get("llm_model")
    q1_temp = float(request.form.get("q1_temp", 0))
    q2_temp = float(request.form.get("q2_temp", 0))
    q3_temp = float(request.form.get("q3_temp", 0))
    custom_seed = int(request.form.get("custom_seed", 1234))
    openai_key = request.form.get("openai_key")

    if not is_openai_key_valid(openai_key):
        return jsonify({"error": "Invalid OpenAI API key."}), 400

    topics_raw = request.form.get("topics_to_evaluate", "").strip()
    topics_to_evaluate = [int(t.strip()) for t in topics_raw.split(",") if t.strip().isdigit()] if topics_raw else None

    output_path = Path(task_dir) / "user_provided.json"

    status, tm_model_data_path = proxann.generate_user_provided_json(
        path_user_study_config_file="data/user_study/config_pilot_test.conf",
        user_provided_tpcs=topics_to_evaluate,
        output_path=output_path
    )

    if status != 0:
        return jsonify({"error": "Failed to generate JSON"}), 500

    task_results[task_id] = {"status": "processing", "timestamp": datetime.utcnow()}

    def run_background():
        try:
            df, _ = proxann.run_metric(
                tm_model_data_path.as_posix(),
                llm_models=[llm_model],
                q1_temp=q1_temp,
                q2_temp=q2_temp,
                q3_temp=q3_temp,
                custom_seed=custom_seed,
                openai_key=openai_key,
            )
            
            df['id'] = pd.to_numeric(df['id'], errors='coerce') 
            df.sort_values(by='id', inplace=True)                    
            df['id'] = df['id'].astype(str)
            df.columns = [format_column_label_html(col) for col in df.columns]            
            table_html = df.to_html(escape=False, classes='table table-striped table-bordered text-center', index=False)
            task_results[task_id] = {
                "status": "done",
                "table": table_html,
                "timestamp": datetime.utcnow()
            }
        except Exception as e:
            task_results[task_id] = {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow()
            }
        finally:
            # remove all information about the task after processing
            try:
                shutil.rmtree(task_dir)
                logger.info(f"Deleted task directory: {task_dir}")
            except Exception as e:
                logger.warning(f"Could not delete task directory {task_dir}: {e}")

    threading.Thread(target=run_background).start()
    return jsonify({"task_id": task_id})

@app.route('/evaluate/status/<task_id>', methods=['GET'])
def evaluate_status(task_id):
    result = task_results.get(task_id)
    if not result:
        return jsonify({"status": "not_found"}), 404
    return jsonify(result)

if __name__ == '__main__':
    threading.Thread(target=cleanup_old_tasks, daemon=True).start()
    app.run(host='0.0.0.0', port=8080, debug=True)