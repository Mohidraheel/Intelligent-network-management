
from flask import Flask, request, jsonify, render_template, send_from_directory
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__, static_folder='static', template_folder='templates')

# --- Configuration / sample logs ---
SAMPLE_LOGS = """2024-10-25 10:15:23 [CRITICAL] Router-01: High CPU usage detected - 95%
2024-10-25 10:16:45 [WARNING] Switch-03: Port 24 flapping detected
2024-10-25 10:17:12 [INFO] Firewall-01: Configuration backup completed
2024-10-25 10:18:33 [CRITICAL] Server-DB01: Connection timeout - Database unreachable
2024-10-25 10:19:01 [WARNING] Router-02: BGP session down with peer 192.168.1.1
2024-10-25 10:20:15 [INFO] Switch-01: VLAN 100 added successfully
2024-10-25 10:21:47 [CRITICAL] Load-Balancer: Health check failed for backend servers
2024-10-25 10:22:03 [WARNING] Router-01: Memory usage high - 87%
2024-10-25 10:23:29 Router-02:High CPU usage detected - 92%"""

# In-memory storage for parsed logs
parsed_logs = []

# Minimal training data for the NB classifier
train_texts = [
    "High CPU usage detected", "Memory usage high", "Health check failed", "Connection timeout",
    "Port flapping detected", "BGP session down", "Configuration backup completed", "VLAN added successfully"
]
train_labels = [
    "CRITICAL", "WARNING", "CRITICAL", "CRITICAL",
    "WARNING", "WARNING", "INFO", "INFO"
]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)
clf = MultinomialNB()
clf.fit(X_train, train_labels)


# --- Parsing / helper functions ---
def parse_logs(log_text):
    logs = []
    # Accept either full logs with [SEVERITY] or fallback to ML classifier
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(.*?)\] (.*?): (.*)'

    for line in log_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if match:
            timestamp, severity, device, message = match.groups()
        else:
            # Try to split "device: message" first
            parts = line.split(':', 1)
            if len(parts) == 2:
                device = parts[0].strip()
                message = parts[1].strip()
            else:
                device = 'Unknown'
                message = line
            X_new = vectorizer.transform([message])
            severity = clf.predict(X_new)[0]
            timestamp = 'Unknown'

        logs.append({
            'timestamp': timestamp,
            'severity': severity,
            'device': device,
            'message': message
        })
    return logs


def count_by_severity(logs):
    counts = {'CRITICAL': 0, 'WARNING': 0, 'INFO': 0}
    for log in logs:
        if log['severity'] in counts:
            counts[log['severity']] += 1
    return counts


def get_critical_logs(logs):
    return [log for log in logs if log['severity'] == 'CRITICAL']


# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/sample', methods=['GET'])
def get_sample():
    return jsonify({'logs': SAMPLE_LOGS})


@app.route('/api/parse', methods=['POST'])
def parse():
    global parsed_logs
    data = request.get_json() or {}
    log_text = data.get('logs', '')

    if not log_text:
        return jsonify({'error': 'No logs provided'}), 400

    parsed_logs = parse_logs(log_text)
    return jsonify({
        'success': True,
        'total': len(parsed_logs),
        'classified': True
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Accept a .txt file uploaded from the web UI. The endpoint parses file content and stores parsed_logs in memory."""
    global parsed_logs
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    # only accept .txt
    if not f.filename.lower().endswith('.txt'):
        return jsonify({'error': 'Only .txt files are accepted'}), 400
    try:
        content = f.read().decode('utf-8')
    except Exception:
        # fallback: try reading bytes directly
        content = f.read().decode('latin-1')

    parsed_logs = parse_logs(content)
    return jsonify({'success': True, 'total': len(parsed_logs)})


@app.route('/api/summary', methods=['GET'])
def summary():
    if not parsed_logs:
        return jsonify({'error': 'No logs parsed yet'}), 400

    severity_counts = count_by_severity(parsed_logs)
    critical_logs = get_critical_logs(parsed_logs)

    return jsonify({
        'total': len(parsed_logs),
        'critical': severity_counts['CRITICAL'],
        'warning': severity_counts['WARNING'],
        'info': severity_counts['INFO'],
        'critical_logs': critical_logs
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    if not parsed_logs:
        return jsonify({'message': 'Please load logs first'})

    data = request.get_json() or {}
    user_input = data.get('message', '').lower()

    if 'critical' in user_input:
        critical = get_critical_logs(parsed_logs)
        return jsonify({'message': f'Found {len(critical)} critical alerts', 'data': critical})
    elif 'stats' in user_input or 'summary' in user_input:
        counts = count_by_severity(parsed_logs)
        return jsonify({
            'message': f"Total: {len(parsed_logs)}, Critical: {counts['CRITICAL']}, Warning: {counts['WARNING']}, Info: {counts['INFO']}"
        })
    elif 'help' in user_input:
        return jsonify({'message': 'Try asking: "show critical", "show stats", or "help"'})
    else:
        return jsonify({'message': 'I can help with: critical alerts, stats, or help'})


# Serve static files (optional)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    print("🚀 ML-powered Network Management Server running at http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)





