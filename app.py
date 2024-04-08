from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('dummyFrontend.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        file_size = len(file.read()) / (1024 * 1024) # Convert to MB
        if file_size <= 50:
            file_name = file.filename
            # Save the file or perform any desired operations
            return jsonify({'fileName': file_name})
        else:
            return jsonify({'error': 'File size exceeds the limit (50 MB).'}), 400
    else:
        return jsonify({'error': 'No file uploaded.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
