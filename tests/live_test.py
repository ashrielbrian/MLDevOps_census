import os
import requests
import json

ENDPOINT = 'http://localhost:8000/'
CSV_SAMPLE_PATH = os.path.join(os.path.dirname(__file__), 'sample.csv')
JSON_SAMPLE_PATH = os.path.join(os.path.dirname(__file__), 'sample.json')


def request_get():
    r = requests.get(ENDPOINT)
    print(r.status_code)


def request_json():
    headers = {'Content-Type': 'application/json'}
    r = requests.post(ENDPOINT + 'inference',
                      json=json.load(open(JSON_SAMPLE_PATH, 'r')),
                      headers=headers)
    print(r.status_code)
    print(r.json())


def request_csv():
    csv_file = {'csv_file': open(CSV_SAMPLE_PATH, 'rb')}
    r = requests.post(ENDPOINT + 'batch_inference', files=csv_file)
    print(r.status_code)
    print(r.json())


if __name__ == '__main__':
    request_get()
    request_csv()
    request_json()
