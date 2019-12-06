from flask import Flask, flash, render_template, redirect, request, send_file, url_for
from extract import extractDate
import os

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = '/media/davinci/New Volume/dontDelete/myWork/fyle/uploads'

@app.route("/", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":
        if request.files:
            image_file = request.files["image"]
            if image_file:
              passed = False
            try:
              filename = image_file.filename
              filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
              image_file.save(filepath)
              passed = True
              print('No exception')

            except Exception:
              print('Hi, im exception')
              passed = False
            if passed:
              return redirect(url_for('function', file_path = filepath ))
    return render_template("upload_image.html")

@app.route('/date')
def function():
  file_path = request.args.get('file_path')
  date=extractDate(file_path)
  return render_template(
        'date.html', date = date
    )

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)