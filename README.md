# Gating Wearable Cognitive Assistant Application

Server and client code for running wearable cognitive assistance applications
with on-device thumbs-up gating created using OpenWorkflow Editor.

Object detection is performed using Ultralytics YOLOv8.

## Creating Application

1. Create an application using the web tool at
   https://cmusatyalab.github.io/OpenWorkflow/
2. Every processor that you add must be of type "YoloProcessor". You can
   import the file `burger.pbfsm`, from this repository, into the web tool to
   see an example.
3. The `model_path` must be a valid path to a trained YOLO model `xxx.pt`.
4. `conf_threshold` is the minimum confidence score we require from the object
   detector. Any bounding boxes with a lower confidence score will be ignored.

## Setting up Zoom

1. Create a Zoom account. On https://zoom.us/profile, you can
   find your personal meeting number and your meeting password from your invite link
   URL, e.g. `https://us06web.zoom.us/j/<MEETING_ID>?pwd=<MEETING_PASSWORD>`.
2. Build an app on https://marketplace.zoom.us/. You can find your Client ID and
   Client Secret on the "Basic Information" page.
3. Continue from above, navigate to "Features" > "Embed", turn on "Embed Meeting SDK
   and bring Zoom features to your app." Then download the zip file of the Zoom SDK
   for Android. Unzip the file, find and copy `mobilertc.aar` to the `android-client/mobilertc/`
   directory of this repository.
4. Create a file called `credentials.py` in the `server` directory of this
   repository. Format it as follows, with the proper values. For example,
   replace "Client ID" with your actual Client ID:
```
CLIENT_ID = 'Client ID'
CLIENT_SECRET = 'Client Secret'

MEETING_NUMBER = 'MEETING_ID'
MEETING_PASSWORD = 'MEETING_PASSWORD'
USER_EMAIL = 'youremail@hostname'
```
5. Go to https://zoom.us/profile/setting. Under "Meeting" > "Schedule Meeting", turn
   on the Participants Video option.
6. Go to https://zoom.us/meeting/schedule. On the "Meeting ID" line, select the radio
   button that starts with "Personal Meeting ID," uncheck "Waiting Room," and set the
   Participant Video option to "on." Then click "Save."

## Installation

1. Set up an SSL certificate for your server using https://letsencrypt.org/
2. Run the following commands to make copies of the credentials:
```
sudo cp /etc/letsencrypt/live/YOUR_HOSTNAME/privkey.pem /path/to/this/repository/server/keys
sudo cp /etc/letsencrypt/live/YOUR_HOSTNAME/fullchain.pem /path/to/this/repository/server/keys
sudo chown $USER /path/to/this/repository/server/keys/privkey.pem
sudo chown $USER /path/to/this/repository/server/keys/fullchain.pem
```
3. Create a Python3.8 virtual environment and install poetry if you have not done so.
```
python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install poetry
```
Then install dependencies for this project:
```
cd /path/to/this/repository/
poetry install
```
4. Start the WCA server:
```
cd /path/to/this/repository/server
python server.py /path/to/your/app.pbfsm
```
Note that the trained YOLO model must be accessible in the
directories that you specified in the web editor.
5. Load the interface for the human expert in a browser by navigating to
   https://<YOUR\_SERVER\_HOSTNAME>:8443/. Note that the page will not load if you do not
   include https at the start of the url.

## Client

1. Add the line `gabrielHost="<YOUR_SERVER_HOSTNAME>"` to
   `android-client/local.properties`
2. Run the client with Android Studio

## Protobuf

`server/wca_state_machine_pb2.py` was copied from
<https://github.com/cmusatyalab/OpenWorkflow/blob/master/gabrieltool/statemachine/wca_state_machine_pb2.py>

`server/wca_pb2.py` gets generated when the Android client is run.
