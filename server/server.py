import ast
import argparse
import json
import logging
import io
import os
import shutil
from datetime import datetime
from collections import namedtuple
from multiprocessing import Process, Pipe

# import cv2
# import numpy as np
from PIL import Image

# import tensorflow as tf
# from object_detection.utils import label_map_util

# import torch
# from torchvision import transforms

from ultralytics import YOLO

from gabriel_server import cognitive_engine
from gabriel_server import local_engine
from gabriel_protocol import gabriel_pb2

import credentials
import http_server
import mpncov
import wca_pb2
import wca_state_machine_pb2

SOURCE = 'wca_client'
INPUT_QUEUE_MAXSIZE = 60
PORT = 9099
NUM_TOKENS = 1
DETECTOR_ONES_SIZE = (1, 480, 640, 3)
CLASSIFIER_THRESHOLD = 0.5

MAX_FRAMES_CACHED = 1000
CACHE_BASEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
DEFAULT_CACHE_DIR = os.path.join(CACHE_BASEDIR, 'last_run')

ALWAYS = 'Always'
HAS_OBJECT_CLASS = 'HasObjectClass'
CLASS_NAME = 'class_name'
TWO_STAGE_PROCESSOR = 'TwoStageProcessor'
GATED_TWO_STAGE_PROCESSOR = 'GatedTwoStageProcessor'
CLASSIFIER_PATH = 'classifier_path'
DETECTOR_PATH = 'detector_path'
DETECTOR_CLASS_NAME = 'detector_class_name'
CONF_THRESHOLD = 'conf_threshold'
THUMBS_UP_REQUIRED = 'thumbs_up_required'
TRANSITION_WORD = 'transition_word'  # Not implemented

LABELS_FILENAME = 'classes.txt'
CLASSIFIER_FILENAME = 'model_best.pth.tar'
LABEL_MAP_FILENAME = 'label_map.pbtxt'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_State = namedtuple('_State', ['always_transition', 'has_class_transitions', 'processors'])
# _Classifier = namedtuple('_Classifier', ['model', 'labels'])
# _Detector = namedtuple('_Detector', ['detector', 'category_index'])

aruco_error_audio = 'Please place the bolt near the aruco marker, and make sure ' \
                    'the marker is fully shown.'
length_error_audio = 'This seems to be a bolt with the incorrect length. ' \
                     'Please put it away and find a 12 millimeter bolt again.'


class _StatesModels:
    def __init__(self, fsm_file_path):
        self._states = {}
        # self._classifiers = {}
        self._object_detectors = {}

        self._classifier_representation = {
            'function': mpncov.MPNCOV,
            'iterNum': 5,
            'is_sqrt': True,
            'is_vec': True,
            'input_dim': 2048,
            'dimension_reduction': None,
        }

        pb_fsm = wca_state_machine_pb2.StateMachine()
        with open(fsm_file_path, 'rb') as f:
            pb_fsm.ParseFromString(f.read())

        for state in pb_fsm.states:
            for processor in state.processors:
                self._load_models(processor)

            assert (state.name not in self._states), 'duplicate state name'
            always_transition = None
            has_class_transitions = {}

            for transition in state.transitions:
                assert (len(transition.predicates) == 1), 'bad transition'

                predicate = transition.predicates[0]
                if predicate.callable_name == ALWAYS:
                    always_transition = transition
                    break

                assert predicate.callable_name == HAS_OBJECT_CLASS, 'bad callable'
                callable_args = json.loads(predicate.callable_args)
                class_name = callable_args[CLASS_NAME]

                has_class_transitions[class_name] = transition

            self._states[state.name] = _State(
                always_transition=always_transition,
                has_class_transitions=has_class_transitions,
                processors=state.processors)

        self._start_state = self._states[pb_fsm.start_state]

    def _load_models(self, processor):
        # TODO: Support YOLO_OD_PROCESSOR
        assert processor.callable_name in [TWO_STAGE_PROCESSOR, GATED_TWO_STAGE_PROCESSOR], \
            'bad processor'
        callable_args = json.loads(processor.callable_args)

        # classifier_dir = callable_args[CLASSIFIER_PATH]
        # if classifier_dir not in self._classifiers:
        #     labels_file = open(os.path.join(classifier_dir, LABELS_FILENAME))
        #     labels = ast.literal_eval(labels_file.read())
        #
        #     freezed_layer = 0
        #     model = mpncov.Newmodel(self._classifier_representation.copy(),
        #                             len(labels), freezed_layer)
        #     model.features = torch.nn.DataParallel(model.features)
        #     model.cuda()
        #     trained_model = torch.load(os.path.join(classifier_dir,
        #                                             CLASSIFIER_FILENAME))
        #     model.load_state_dict(trained_model['state_dict'])
        #     model.eval()
        #
        #     self._classifiers[classifier_dir] = _Classifier(
        #         model=model, labels=labels)

        detector_dir = callable_args[DETECTOR_PATH]

        if detector_dir not in self._object_detectors:
            # detector = tf.saved_model.load(detector_dir)
            # ones = tf.ones(DETECTOR_ONES_SIZE, dtype=tf.uint8)
            # detector(ones)
            #
            # label_map_path = os.path.join(detector_dir, LABEL_MAP_FILENAME)
            # label_map = label_map_util.load_labelmap(label_map_path)
            # categories = label_map_util.convert_label_map_to_categories(
            #     label_map,
            #     max_num_classes=label_map_util.get_max_label_map_index(
            #         label_map),
            #     use_display_name=True)
            # category_index = label_map_util.create_category_index(categories)

            detector = YOLO(detector_dir)
            print(detector.names)
            detector.to('cuda')

            self._object_detectors[detector_dir] = detector

    # def get_classifier(self, path):
    #     return self._classifiers[path]

    def get_object_detector(self, path):
        return self._object_detectors[path]

    def get_state(self, name):
        return self._states[name]

    def get_start_state(self):
        return self._start_state


class _StatesForExpertCall:
    def __init__(self, transition, states_models):
        self._added_states = set()
        self._state_names = []
        self._transition_to_state = {}

        self._states_models = states_models

        if not os.path.exists(http_server.IMAGES_DIR):
            os.mkdir(http_server.IMAGES_DIR)

        self._add_descendants(transition)

    def _add_descendants(self, transition):
        if transition.next_state in self._added_states:
            return

        self._added_states.add(transition.next_state)
        self._state_names.append(transition.next_state)
        self._transition_to_state[transition.next_state] = transition

        img_filename = os.path.join(
            http_server.IMAGES_DIR, '{}.jpg'.format(transition.next_state))
        with open(img_filename, 'wb') as f:
            f.write(transition.instruction.image)

        next_state = self._states_models.get_state(transition.next_state)
        if next_state.always_transition is not None:
            self._add_descendants(next_state.always_transition)
            return

        for transition in next_state.has_class_transitions.values():
            self._add_descendants(transition)

    def get_state_names(self):
        return self._state_names

    def get_transition(self, name):
        return self._transition_to_state[name]


class InferenceEngine(cognitive_engine.Engine):

    def __init__(self, fsm_file_path):
        # ############################################
        # TODO: Add to the protobuf message to make the server stateless
        self._frame_tx_count = 0
        # ############################################
        # physical_devices = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # self._transform = transforms.Compose([
        #     transforms.Resize((448, 448)),
        #     transforms.ToTensor(),
        #     normalize,
        # ])
        self._fsm_file_name = os.path.basename(fsm_file_path)
        self._states_models = _StatesModels(fsm_file_path)

        start_state = self._states_models.get_start_state()
        assert start_state.always_transition is not None, 'bad start state'
        self._states_for_expert_call = _StatesForExpertCall(
            start_state.always_transition,
            self._states_models)
        state_names = self._states_for_expert_call.get_state_names()

        http_server_conn, self._engine_conn = Pipe()
        self._http_server_process = Process(
            target=http_server.start_http_server,
            args=(http_server_conn, state_names))
        self._http_server_process.start()

        self._on_zoom_call = False

        self._frames_cached = []
        if os.path.exists(DEFAULT_CACHE_DIR):
            if os.path.isdir(DEFAULT_CACHE_DIR):
                shutil.rmtree(DEFAULT_CACHE_DIR)
            else:
                os.remove(DEFAULT_CACHE_DIR)
        os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)

    def _result_wrapper_for_transition(self, transition):
        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)

        logger.info('sending %s', transition.instruction.audio)

        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.TEXT
        result.payload = transition.instruction.audio.encode()
        result_wrapper.results.append(result)

        if len(transition.instruction.image) > 0:
            result = gabriel_pb2.ResultWrapper.Result()
            result.payload_type = gabriel_pb2.PayloadType.IMAGE
            result.payload = transition.instruction.image
            result_wrapper.results.append(result)

        if len(transition.instruction.video) > 0:
            result = gabriel_pb2.ResultWrapper.Result()
            result.payload_type = gabriel_pb2.PayloadType.VIDEO
            result.payload = transition.instruction.video
            result_wrapper.results.append(result)

        to_client_extras = wca_pb2.ToClientExtras()
        to_client_extras.step = transition.next_state
        to_client_extras.zoom_result = wca_pb2.ToClientExtras.ZoomResult.NO_CALL

        assert transition.next_state != '', "invalid transition end state"
        to_client_extras.user_ready = wca_pb2.ToClientExtras.UserReady.DISABLE
        next_processors = self._states_models.get_state(transition.next_state).processors
        if len(next_processors) == 0:
            # End state reached
            logger.info("Client done. # Frame transmitted = %s", self._frame_tx_count)
            to_client_extras.step = "WCA_FSM_END"

        result_wrapper.extras.Pack(to_client_extras)
        return result_wrapper

    def _result_wrapper_for(self,
                            step,
                            zoom_result=wca_pb2.ToClientExtras.ZoomResult.NO_CALL,
                            audio=None,
                            user_ready=wca_pb2.ToClientExtras.UserReady.NO_CHANGE):
        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        to_client_extras = wca_pb2.ToClientExtras()
        to_client_extras.step = step
        to_client_extras.zoom_result = zoom_result
        to_client_extras.user_ready = user_ready

        if audio is not None:
            result = gabriel_pb2.ResultWrapper.Result()
            result.payload_type = gabriel_pb2.PayloadType.TEXT
            result.payload = audio.encode()
            result_wrapper.results.append(result)

        result_wrapper.extras.Pack(to_client_extras)
        return result_wrapper

    def _start_zoom(self):
        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        to_client_extras = wca_pb2.ToClientExtras()
        to_client_extras.zoom_result = wca_pb2.ToClientExtras.ZoomResult.CALL_START
        to_client_extras.user_ready = wca_pb2.ToClientExtras.UserReady.NO_CHANGE

        zoom_info = wca_pb2.ZoomInfo()
        zoom_info.jwt_token = http_server.gen_jwt_token(credentials.CLIENT_ID, credentials.CLIENT_SECRET,
                                                        credentials.MEETING_NUMBER, http_server.ROLE_PARTICIPANT)
        zoom_info.meeting_number = credentials.MEETING_NUMBER
        zoom_info.meeting_password = credentials.MEETING_PASSWORD

        to_client_extras.zoom_info.CopyFrom(zoom_info)

        result_wrapper.extras.Pack(to_client_extras)
        return result_wrapper

    def _try_start_zoom(self, step):
        if self._on_zoom_call:
            return self._result_wrapper_for(step,
                                            zoom_result=wca_pb2.ToClientExtras.ZoomResult.EXPERT_BUSY)
        msg = {
            'zoom_action': 'start',
            'step': step
        }
        self._engine_conn.send(msg)
        logger.info('Sending Zoom info to client.')
        return self._start_zoom()

    def handle(self, input_frame):
        to_server_extras = cognitive_engine.unpack_extras(
            wca_pb2.ToServerExtras, input_frame)
        print('.', end='')

        if (to_server_extras.client_cmd ==
                wca_pb2.ToServerExtras.ClientCmd.ZOOM_STOP):
            msg = {
                'zoom_action': 'stop'
            }
            # TODO: Fix bug: When a client join the Zoom multiple times, it only sees the web-user-set steps
            #       returned from the expert at the first time
            self._engine_conn.send(msg)
            pipe_output = self._engine_conn.recv()
            new_step = pipe_output.get('step')
            logger.info('Zoom Stopped. New step: %s', new_step)
            transition = self._states_for_expert_call.get_transition(new_step)
            return self._result_wrapper_for_transition(transition)

        step = to_server_extras.step
        if step == "WCA_FSM_START" or not step:
            state = self._states_models.get_start_state()
            self._frame_tx_count = 0
        elif step == "WCA_FSM_END":
            return self._result_wrapper_for(step,
                                            user_ready=wca_pb2.ToClientExtras.UserReady.DISABLE)
        elif (to_server_extras.client_cmd ==
              wca_pb2.ToServerExtras.ClientCmd.ZOOM_START):
            return self._try_start_zoom(step)
        else:
            state = self._states_models.get_state(step)

        self._frame_tx_count += 1

        # Save current cache folder and create a new cache folder
        if (to_server_extras.client_cmd ==
                wca_pb2.ToServerExtras.ClientCmd.REPORT):
            report_time = datetime.now().strftime('-%Y-%m-%d-%H-%M-%S-%f')
            log_dirname = self._fsm_file_name.split('.')[0] + report_time
            os.rename(DEFAULT_CACHE_DIR, os.path.join(CACHE_BASEDIR, log_dirname))
            os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)

        if state.always_transition is not None:
            return self._result_wrapper_for_transition(state.always_transition)

        assert len(state.processors) == 1, 'wrong number of processors'
        processor = state.processors[0]

        callable_args = json.loads(processor.callable_args)
        detector_dir = callable_args[DETECTOR_PATH]
        detector = self._states_models.get_object_detector(detector_dir)

        if not input_frame.payloads:
            return self._result_wrapper_for(step)
        # np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        # img_bgr = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        #
        # # Insert a new axis to make an input tensor of shape (1, h, w, channel)
        # # For Google Glass, the input shape is (1, 1080, 1920, 3)
        # detections = detector.detector(np.expand_dims(img, 0))
        # scores = detections['detection_scores'][0].numpy()
        # boxes = detections['detection_boxes'][0].numpy()
        # classes = detections['detection_classes'][0].numpy().astype(int)
        #
        # im_height, im_width = img.shape[:2]
        #
        # classifier_dir = callable_args[CLASSIFIER_PATH]
        # classifier = self._states_models.get_classifier(classifier_dir)

        pil_img = Image.open(io.BytesIO(input_frame.payloads[0]))

        conf_threshold = float(callable_args[CONF_THRESHOLD])
        detector_class_name = callable_args[DETECTOR_CLASS_NAME]

        detection_result = detector(pil_img, conf=conf_threshold, verbose=False)[0]
        good_boxes = []
        box_scores = []
        # for score, box, class_id in zip(scores, boxes, classes):
        #     class_name = detector.category_index[class_id]['name']
        #     if score > conf_threshold and class_name == detector_class_name:

        for box in detection_result.boxes:
            class_id = int(box.cls)
            class_name = detection_result.names[class_id]
            # Make sure the classnames in YOLO matches with those in FSM
            if detector_class_name in class_name:
                bi = 0
                while bi < len(box_scores):
                    if box.conf > box_scores[bi]:
                        break
                    bi += 1
                good_boxes.insert(bi, box)
                box_scores.insert(bi, box.conf)

        # Cache the current frame
        if len(self._frames_cached) >= MAX_FRAMES_CACHED:
            frame_to_evict = self._frames_cached.pop(0)
            try:
                os.remove(frame_to_evict)
            except OSError as oe:
                logger.warning(oe)
        cur_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f-')
        detected_class = detector_class_name if good_boxes else 'none'
        cached_filename = os.path.join(DEFAULT_CACHE_DIR,
                                       cur_time + detected_class + '(' + detector_class_name + ').jpg')
        self._frames_cached.append(cached_filename)
        pil_img.save(cached_filename)

        if not good_boxes:
            return self._result_wrapper_for(step)

        print()
        print('Detector boxes:', box_scores)
        for best_box in good_boxes:
            # ymin, xmin, ymax, xmax = best_box
            # (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
            #                               ymin * im_height, ymax * im_height)
            #
            # cropped_pil = pil_img.crop((left, top, right, bottom))
            # transformed = self._transform(cropped_pil).cuda()
            # output = classifier.model(transformed[None, ...])
            # prob = torch.nn.functional.softmax(output, dim=1)
            # # print('Classifier probability:', prob.data.cpu().numpy())
            #
            # value, pred = prob.topk(1, 1, True, True)
            # if value.item() < CLASSIFIER_THRESHOLD:
            #     continue
            # class_ind = pred.item()
            # label_name = classifier.labels[class_ind]
            #
            # # Cache the cropped frame
            # if len(self._frames_cached) >= MAX_FRAMES_CACHED:
            #     frame_to_evict = self._frames_cached.pop(0)
            #     try:
            #         os.remove(frame_to_evict)
            #     except OSError as oe:
            #         logger.warning(oe)
            # classified_class = label_name
            # cached_filename = os.path.join(DEFAULT_CACHE_DIR,
            #                                cur_time + "cropped-" + classified_class + '.jpg')
            # self._frames_cached.append(cached_filename)
            # cropped_pil.save(cached_filename)
            #
            # logger.info('Found label: %s', label_name)
            # print('Classifier probability:', value.item())
            # # logger.info('return transition: %s', str(state.has_class_transitions.keys()))
            # # logger.info('current state name on server is: %s', step)

            label_name = detected_class

            if label_name is not None:
                transition = state.has_class_transitions.get(label_name)
                if transition is not None:
                    return self._result_wrapper_for_transition(transition)
            return self._result_wrapper_for(step)

        # Good boxes found but not classified as any valid steps
        return self._result_wrapper_for(step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fsm_file_path', type=str)
    args = parser.parse_args()

    def engine_factory():
        return InferenceEngine(args.fsm_file_path)

    local_engine.run(
        engine_factory, SOURCE, INPUT_QUEUE_MAXSIZE, PORT, NUM_TOKENS)


if __name__ == '__main__':
    main()
