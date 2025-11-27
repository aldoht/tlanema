import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os
import tensorflow as tf
import time

class OutsideOptionRange(Exception):
    pass

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (255, 0, 0) # red

CLASSES_MAP = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'O': 12,
    'P': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'U': 17,
    'V': 18,
    'W': 19,
    'Y': 20
}
INVERSE_CLASSES_MAP = {v: k for k, v in CLASSES_MAP.items()}

landmarks_map = {
    0: 'WRIST',
    1: 'THUMB_CMC',
    2: 'THUMB_MCP',
    3: 'THUMB_IP',
    4: 'THUMB_TIP',
    5: 'INDEX_FINGER_MCP',
    6: 'INDEX_FINGER_PIP',
    7: 'INDEX_FINGER_DIP',
    8: 'INDEX_FINGER_TIP',
    9: 'MIDDLE_FINGER_MCP',
    10: 'MIDDLE_FINGER_PIP',
    11: 'MIDDLE_FINGER_DIP',
    12: 'MIDDLE_FINGER_TIP',
    13: 'RING_FINGER_MCP',
    14: 'RING_FINGER_PIP',
    15: 'RING_FINGER_DIP',
    16: 'RING_FINGER_TIP',
    17: 'PINKY_MCP',
    18: 'PINKY_PIP',
    19: 'PINKY_DIP',
    20: 'PINKY_TIP'
}

model_dict = {
    0: ('./lsm-detector-hands-mobilenetv2-61e.keras', False, 3),
    1: ('./lsm-detector-hands-scratch-84e.keras', True, 1)
}

# HandLandmarker object
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path='./hand_landmarker.task'
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

def open_cam() -> (cv2.VideoCapture, int, int):
    cam = cv2.VideoCapture(0)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cam, frame_width, frame_height

def close_cam(camera: cv2.VideoCapture) -> None:
    camera.release()
    cv2.destroyAllWindows()
    return

# Returns handmarks information in image at img_path
def detect_img(image: mp.Image):
    original_detection = detector.detect(image)
    flipped_detection = original_detection

    # Handle empty detection
    if not original_detection.handedness:
        return -1,-1,-1
  
    # Handle left-handed image
    if original_detection.handedness[0][0].display_name == 'Left':
        flipped = cv2.flip(image.numpy_view(), 1)
        image = mp.Image(image_format=image.image_format, data=flipped)
        flipped_detection = detector.detect(image)

        # Handle empty detection after flipping
        if not flipped_detection.handedness:
            return -1,-1,-1

    return original_detection, image, flipped_detection

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    hand_landmarks = hand_landmarks_list[0]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style()
    )

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    return annotated_image, text_x, text_y

def get_hand_from_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    cropped_image = np.copy(rgb_image)

    hand_landmarks = hand_landmarks_list[0]

    # Get the hand landmarks from detection
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])

    # Separate the hand landmarks (x,y,z) -> x, (x,y,z) -> y
    height, width, channels = cropped_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]

    # Crop the image
    crop_margin = 70

    start_x = int(min(x_coordinates) * width) - crop_margin if (int(min(x_coordinates) * width) - crop_margin) > 0 else 0
    start_y = (int(min(y_coordinates) * height) - crop_margin) if (int(min(y_coordinates) * height) - crop_margin) > 0 else 0

    end_x = int(max(x_coordinates) * width) + crop_margin if (int(max(x_coordinates) * width) + crop_margin) < width else width
    end_y = int(max(y_coordinates) * height) + crop_margin if (int(max(y_coordinates) * height) + crop_margin) < height else height

    return cropped_image[start_y:end_y, start_x:end_x]

def process_hand_image(image, greyscale=True, height=60, width=60):
    _, image, detection = detect_img(image)

    # Empty detection
    if detection == -1:
        print('No hand was detected.')
        return None

    handImage = get_hand_from_image(image.numpy_view(), detection)

    if greyscale:
        handImage = cv2.cvtColor(cv2.resize(handImage,(height,width)), cv2.COLOR_RGB2GRAY)
        handImage = np.expand_dims(handImage, axis=-1)
    else:
        handImage = cv2.resize(handImage,(height,width))
        handImage = cv2.cvtColor(handImage, cv2.COLOR_BGR2RGB)

    print(f'Image was processed.')

    return image, handImage

def map_prediction_to_label(pred_array: np.array) -> str:
    index = np.argmax(pred_array, axis=1)[0]
    return INVERSE_CLASSES_MAP[index]

def select_model() -> tuple[str, bool, int]:
    while True:
        print('Seleccione el modelo a utilizar: ')
        for model_idx, model_path in model_dict.items():
            print(f'{model_idx}: {model_path[0][2:]}', end='\n')
        print('Modelo:', end=' ')
        try:
            user_idx = int(input())
            if not (user_idx in model_dict.keys()):
                raise OutsideOptionRange
            return model_dict[user_idx]
        except ValueError:
            print('No se ingreso un numero.', end=' ')
        except OutsideOptionRange:
            print(f'El numero {user_idx} no esta en las opciones.', end=' ')
        print('Intente de nuevo por favor.')

model_path, greyscale_value, channel_size = select_model()
if not os.path.exists(model_path):
    print(f'Model was not found in path {model_path}.')
else:
    try:
        loaded_cnn = tf.keras.models.load_model(model_path, compile=True)
        print(f'Model was loaded correctly.')
    except Exception as e:
        print(f'Error when loading the model in {model_path}: {e}.')
        quit()

cam, w, h = open_cam()

secondsBetweenPredictions = 0.5
last_prediction_time = 0
last_prediction_text = ""

while True:
    ret, frame = cam.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    original_detection_result, image, _ = detect_img(mp_image)

    current_time = time.time()

    # No hand detected
    if original_detection_result == -1:
        cv2.imshow('Camera', cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR))
    else:
        annotated_image, x, y = draw_landmarks_on_image(mp_image.numpy_view(), original_detection_result)

        if current_time - last_prediction_time >= secondsBetweenPredictions:
            test_original, test_hand = process_hand_image(image, greyscale=greyscale_value, height=224, width=224)
            test_hand_reshaped = test_hand.reshape(1, 224, 224, channel_size)
            prediction = loaded_cnn.predict(test_hand_reshaped)
            last_prediction_text = f"{map_prediction_to_label(prediction)}: {np.max(prediction)*100:.1f}%"
            last_prediction_time = current_time

        cv2.putText(annotated_image, last_prediction_text,
                    (x, y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        cv2.imshow('Camera', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User quit.")
        break