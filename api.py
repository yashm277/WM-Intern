# add data on the fly
# 4 apis - import and train - get a picklefile - blob object, import and retrain, predict for designated model, load model also takes blob.
import os, json
from flask import Flask, flash, request, redirect, url_for
from runThis import trainingEncapsulated, final
UPLOAD_FOLDER = '/Users/yashmundada/Desktop/service-virtualisation-final/data'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.secret_key = "super secret key"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/import_and_train', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("hello!!")
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print("I'm working I need some time!")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            print("I've saved your file")
            # print(os.path.join(app.config['UPLOAD_FOLDER'])+"/"+file.filename)
            # print(file.filename)
            trainingEncapsulated(os.path.join(app.config['UPLOAD_FOLDER'])+"/"+file.filename, file.filename)
            print("Trained!")
            return '''
            <!doctype html>
            <h1>Successfully uploaded and trained</h1>  
            </form>
            '''
    return '''
    <!doctype html>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def get_model_options():
    models_folder = "/Users/yashmundada/Desktop/service-virtualisation-final/models/"
    files = os.listdir(models_folder)
    model_options = [os.path.splitext(file)[0] for file in files]
    return model_options

def get_csv_options():
    csv_folder = "/Users/yashmundada/Desktop/service-virtualisation-final/data/"
    files = os.listdir(csv_folder)
    csv_options = [os.path.splitext(file)[0] for file in files]
    return csv_options

from flask import send_from_directory

@app.route('/download_csv/<name>', methods=['GET', 'POST'])
def download_csv_(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name+".csv")
# @app.route('/download_model/<name>', methods=['GET', 'POST'])
# def download_model_pickle(name):
#     return send_from_directory("/Users/yashmundada/Desktop/service-virtualisation-final/models", name+".pkl")

@app.route('/download_csv', methods=['GET', 'POST'])
def download_csv_file():
    if request.method == 'POST':
        csv_file = request.form['csv']
        return redirect(url_for('download_csv_file', name=csv_file))
    csv_options = get_csv_options()
    options_html = "\n".join(f'<option value="{option}">{option}</option>' for option in csv_options)
    return f'''
<!doctype html>
<html>
<head>
    <title>Dropdown Menu</title>
</head>
<body>
    <h1>Choose the csv file to download</h1>
    <form method="post">
        <select name="csv">
            {options_html}
        </select>
        <input type="submit" value="Download">
    </form>
</body>
</html>
'''
@app.route('/download_model', methods=['GET', 'POST'])
def download_model_pickle(name):
    return send_from_directory("/Users/yashmundada/Desktop/service-virtualisation-final/models", name+".pkl")

    
@app.route('/')
def hello_world():
    return ''''
    <!doctype html>
    <h1>Welcome to the opaqueSV service</h1>
    <form method="post">
    <input type="button" value="Import and Train" onclick="window.location.href='/import_and_train'">
    <input type="button" value="Import and Retrain" onclick="window.location.href='/import_and_retrain'">
    <input type="button" value="Predict for a chosen model" onclick="window.location.href='/predict'">
    <input type="button" value="Download CSV" onclick="window.location.href='/download_csv'">
    <input type="button" value="Download Model Pickle" onclick="window.location.href='/download_model'">
    </form>
'''

# # load and retrain
# # load a model, retrain the model and return the retrained model
@app.route('/import_and_retrain', methods=['GET'])
def import_and_retrain():
    return ''''
    <!doctype html>
    <h1>Currently a work in progress...</h1>
    '''
    pass


# # predict
# # predict for a model and return the prediction

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        SVreq = json.loads(request.form['data'])
        selected_model = request.form['model']
        result = final(SVreq, selected_model)        
        json_result = json.dumps(result, indent=4).replace('\\', '<br>')
        
        return '''
        <!doctype html>
        <html>
        <head>
          <title>Successfully predicted</title>
        </head>
        <body>
          <h1>Successfully predicted</h1>
          <p>The prediction is: </p>
          <p>{}<p>
        </body>
        </html>
        '''.format(json_result)
    # fetch the model options from the models folder and display them in the dropdown
    model_options = get_model_options()
    options_html = "\n".join(f'<option value="{option}">{option}</option>' for option in model_options)
    return f'''
<!doctype html>
<html>
<head>
  <title>Dropdown Menu</title>
</head>
<body>
  <h1>Enter your data</h1>
  <form method="post">
    <input type="text" name="data">
    <input type="submit" value="Predict">
    <select name="model">
      {options_html}
    </select>
  </form>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)
    