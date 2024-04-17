import os
import json
import requests
import credentials
from datetime import date, timedelta


API_SERVER = "https://api.zoom.us/v2"
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
CHUNK_SIZE = 1 << 20  # 1 MB

list_recordings_api = "/users/" + credentials.USER_EMAIL + "/recordings"
list_recordings_url = API_SERVER + list_recordings_api
today = date.today().strftime("%Y-%m-%d")
last_month = date.today() - timedelta(days=30)
payload = {'access_token': credentials.JWT_TOKEN, 'from': last_month, 'to': today}


if __name__ == "__main__":
    response = requests.get(list_recordings_url, params=payload)
    meetings_list = []
    try:
        recordings_list = response.json()
        if recordings_list.get('code') is not None:
            print(recordings_list)
            exit(1)
        meetings_list = recordings_list.get('meetings')
    except json.decoder.JSONDecodeError:
        print("Empty response returned. Please check URL again.")
        exit(1)

    downloaded_recordings = os.listdir(RECORDINGS_DIR)
    for meeting in meetings_list:
        recordings = meeting.get('recording_files')
        for file in recordings:
            recording_start = file.get('recording_start').replace(":", "-")
            file_id = file.get('id')
            file_type = file.get('file_extension')
            download_url = file.get('download_url')
            filename = recording_start + file_id + "." + file_type

            if filename not in downloaded_recordings:
                print("Downloading " + filename + "...", end=" ")

                # Referring to https://requests.readthedocs.io/en/latest/user/quickstart/#raw-response-content
                r = requests.get(download_url, stream=True)
                filepath = os.path.join(RECORDINGS_DIR, filename)
                with open(filepath, 'wb') as fi:
                    for chunk in r.iter_content(CHUNK_SIZE):
                        fi.write(chunk)
                print("Done")
