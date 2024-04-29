import os
from flask import Flask, request, render_template, jsonify
# from llm import classify_file, create_llm

app = Flask(__name__)

# llm_created = False

# @app.before_request
# def initialize_llm():
#     global llm_created
#     if not llm_created:
#         create_llm()
#         llm_created = True

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file to a temporary directory
            file_path = os.path.join('uploadsTemp', file.filename)
            file.save(file_path)

            # Call classify_file function with the uploaded file path
            # classified_result = classify_file(file_path)

            # Delete the uploaded file after processing
            os.remove(file_path)

            # Pass the classified result to the HTML template
            return jsonify(file.name)

    return render_template('dummyFrontend.html')  # Render the HTML template for file upload

if __name__ == '__main__':
    app.run(debug=True)
