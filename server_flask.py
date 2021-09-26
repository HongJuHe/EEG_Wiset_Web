from flask import Flask, render_template, request
import os
import numpy as np
import keras
from keras.models import load_model
import json
import datetime as dt
import boto3

app = Flask(__name__)


# main page
@app.route('/')
def main():
    return render_template('main.html')


# map page
@app.route('/map')
def map():
    return render_template('map.html')


# records page
@app.route('/records')
def record():
    send_data = []
    # get depression_result
    depression_data = []
    temp_data = {}
    with open('./static/data/depression_result.json', 'r') as f:
        temp_data = json.load(f)

    for d in temp_data:
        d_data = []
        d_data.append(d)
        d_data.append(temp_data[d]['result'])
        d_data.append(temp_data[d]['percent'])
        depression_data.append(d_data)

    print(depression_data)

    # get intent
    intent = []
    dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-2')
    table = dynamodb.Table('user_database')
    response = table.scan()
    items = response['Items']
    for item in items:
        temp_intent = []
        for n in item['events'][3]['parse_data']['intent_ranking']:
            if n['confidence'] > 0.8:
                temp_intent.append(n['name'])
        if (len(temp_intent) != 0):
            intent.append(temp_intent)

    print(intent)

    send_data.append(depression_data)
    send_data.append(intent)
    return render_template('records.html', value=send_data)


# upload html rendering
@app.route('/send_file')
def render_file():
    return render_template('send_file.html')


# file upload work
@app.route('/fileUpload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        x = dt.datetime.now()
        f_name = str(x.year) + "_" + str(x.month) + "_" + str(x.day) + ".txt"
        path_dir = './static/data/'
        f.save(os.path.join(path_dir, f_name))
        print(f_name)
        # f.save('/home/ubuntu/my_tensorflow/uploads/'+secure_filename(f.filename))
        # temp_list = np.loadtxt(f, dtype='float')
        # temp_string = f.decode()
        # temp_list = np.array(map(float, temp_string.split()))
        # print(temp_list)

        f = open(os.path.join(path_dir, f_name), "r")
        content = f.read()
        content_list = content.split(" ")
        f.close()
        # print(len(content_list))
        temp_list = []
        for i in content_list:
            if i != '':
                temp_list.append(float(i))

        print(len(temp_list))
        data_arr = np.array(temp_list[:210000])
        print(np.shape(data_arr))
        data_arr = np.reshape(data_arr, (1, 500, 420))
        print(np.shape(data_arr))
        model = load_model('eeg_deeplearning_conv1d_lstm.h5')
        result = model.predict(data_arr)

        return render_template('send_result.html', value=result)


if __name__ == '__main__':
    app.debug = True
    app.run()