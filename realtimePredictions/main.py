import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import tensorflow as tf
import time

class OutsideOptionRange(Exception):
    pass

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (255, 0, 0) # red

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

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
    0: ('./realtimePredictions/lsm-detector-hands-mobilenetv2-61e.keras', False, 3),
    1: ('./realtimePredictions/lsm-detector-hands-scratch-84e.keras', True, 1)
}

# HandLandmarker object
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path='./realtimePredictions/hand_landmarker.task'
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

def open_cam() -> tuple[cv2.VideoCapture, int, int]:
    """
    Opens the user's camera.
    
    Args:
        None.
    
    Returns:
        A tuple containing a VideoCapture object along with the camera dimensions (width, height).
    """
    cam = cv2.VideoCapture(0)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cam, frame_width, frame_height

def close_cam(camera: cv2.VideoCapture) -> None:
    """
    Closes the user's camera.
    
    Args:
        camera: A VideoCapture object that was created by the open_cam function.
    
    Returns:
        None.
    """
    camera.release()
    cv2.destroyAllWindows()
    return

def detect_img(image: mp.Image):
    """
    Makes the detection of a hand in an image.
    
    Args:
        image: The image where the detection will take place.
    
    Returns:
        A tuple containing two detections and the original image, or a tuple[-1, -1, -1] if there was an error while detecting the image.
    """
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
    """
    Draws the hand landmarks on an image.
    
    Args:
        rgb_image: The image where the landmarks will be drawn.
        detection_result: The detection performed on rgb_image.
    
    Returns:
        A tuple containing the annotated image and the (x,y) coordinates where text was written.
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    text_x, text_y = 0, 0

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
        annotated_image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image, text_x, text_y

def get_hand_from_image(rgb_image: np.ndarray, detection_result, hand_index: int = 0, crop_margin: int = 70) -> np.ndarray | None:
    """
    Crops a hand region from an image based on MediaPipe hand detection results.

    Args:
        rgb_image: The input RGB image as a NumPy array.
        detection_result: The result from MediaPipe HandLandmarker.
        hand_index: Index of the hand to crop (0 = first detected hand).
        crop_margin: Pixel margin to add around the hand bounding box.

    Returns:
        Cropped NumPy array of the hand region, or None if no hand is found.
    """
    hand_landmarks_list = detection_result.hand_landmarks

    if not hand_landmarks_list or hand_index >= len(hand_landmarks_list):
        return None

    hand_landmarks = hand_landmarks_list[hand_index]
    height, width = rgb_image.shape[:2]

    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]

    start_x = max(0, int(min(x_coordinates) * width) - crop_margin)
    start_y = max(0, int(min(y_coordinates) * height) - crop_margin)
    end_x   = min(width,  int(max(x_coordinates) * width)  + crop_margin)
    end_y   = min(height, int(max(y_coordinates) * height) + crop_margin)

    return np.copy(rgb_image[start_y:end_y, start_x:end_x])

def process_hand_image(image, greyscale=True, height=60, width=60):
    """
    Processes a hand image so it can be fed to the model. Defaults for the greyscale model.
    
    Args:
        image: The original image to perform inference.
        greyscale: Wheter the image will be turn into greyscale or not.
        height: Final height of the image.
        width: Final width of the image.
    
    Returns:
        The original image and its processed hand image.
    
    Raises:
        An error is raised if there is no detected hand in the image.
    """
    _, image, detection = detect_img(image)

    # Empty detection
    if detection == -1 or image == -1:
        print('No hand was detected.')
        return None, None

    handImage = get_hand_from_image(image.numpy_view(), detection)
    if handImage is None:
        raise

    size = (int(width), int(height))
    if greyscale:
        handImage = cv2.cvtColor(cv2.resize(handImage, size), cv2.COLOR_RGB2GRAY)
        handImage = np.expand_dims(handImage, axis=-1)
    else:
        handImage = cv2.resize(handImage, size)
        handImage = cv2.cvtColor(handImage, cv2.COLOR_BGR2RGB)

    print('Image was processed.')

    return image, handImage

def map_prediction_to_label(pred_array: np.ndarray) -> str:
    """
    Maps the model's prediction to a label.
    
    Args:
        pred_array: The np.ndarray that contains the probabilities predicted for each class by the model.
    
    Returns:
        The label of the predicted class.
    """
    index = np.argmax(pred_array, axis=1)[0]
    return INVERSE_CLASSES_MAP[index]

def select_model() -> tuple[str, bool, int]:
    """
    Makes the user select a model.
    
    Args:
        None.
    
    Returns:
        A tuple containing the path, channel, and channel size of the selected model.
    """
    user_idx = -1
    while True:
        print('Seleccione el modelo a utilizar: ')
        for model_idx, model_path in model_dict.items():
            print(f'{model_idx}: {model_path[0][2:]}', end='\n')
        print('Modelo:', end=' ')
        try:
            user_idx = int(input())
            if user_idx not in model_dict.keys():
                raise OutsideOptionRange
            return model_dict[user_idx]
        except ValueError:
            print('No se ingreso un numero.', end=' ')
        except OutsideOptionRange:
            print(f'El numero {user_idx} no esta en las opciones.', end=' ')
        print('Intente de nuevo por favor.')

if __name__ == '__main__':
    model_path, greyscale_value, channel_size = select_model()
    loaded_cnn = None
    
    if not os.path.exists(model_path):
        print(f'Model was not found in path {model_path}.')
        exit(1)
    else:
        try:
            loaded_cnn = tf.keras.models.load_model(model_path, compile=True)
            print('Model was loaded correctly.')
        except Exception as e:
            print(f'Error when loading the model in {model_path}: {e}.')
            exit(1)
    
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
                if test_original is None or test_hand is None:
                    continue
                test_hand_reshaped = test_hand.reshape(1, 224, 224, channel_size)
                prediction = loaded_cnn.predict(test_hand_reshaped)
                last_prediction_text = f"{map_prediction_to_label(prediction)}: {np.max(prediction)*100:.1f}%"
                last_prediction_time = current_time
    
            cv2.putText(annotated_image, last_prediction_text,
                        (x, y + FONT_SIZE*30), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            cv2.imshow('Camera', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User quit.")
            break