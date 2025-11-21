import cv2
import time
import os
import argparse

total_classes = 21
user_dict = {
    1: 'Aldo',
    2: 'Diego',
    3: 'Isaias',
    4: 'Otro'
}

LETTER_TO_LABEL_MAP = {
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
LABEL_TO_LETTER_MAP = {v:k for k,v in LETTER_TO_LABEL_MAP.items()}

class OutsideOptionRange(Exception):
    pass

def open_cam() -> (cv2.VideoCapture, int, int):
    cam = cv2.VideoCapture(0)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cam, frame_width, frame_height

def close_cam(camera: cv2.VideoCapture) -> None:
    camera.release()
    cv2.destroyAllWindows()
    return

def get_user() -> str:
    user_idx = -1
    while True:
        print('Selecciona tu usuario: ')
        for u_idx, user in user_dict.items():
            print(f'{u_idx}: {user}', end='\n')
        print('Usuario:', end=' ')
        try:
            user_idx = int(input())
            if not (user_idx in user_dict.keys()):
                raise OutsideOptionRange
            return ''.join(list(user_dict[user_idx])[0:2])
        except ValueError:
            print('No se ingreso un numero.', end=' ')
        except OutsideOptionRange:
            print(f'El numero {user_idx} no esta en las opciones.', end=' ')
        print('Intente de nuevo por favor.')

def first_menu() -> int:
    option = -1
    while True:
        print('Elija una de las siguientes opciones:',
              '1. Tomar todas las fotos de todas las letras.',
              '2. Tomar todas las fotos de una sola letra.',
              '3. Salir.',
              sep='\n')
        print('Opcion:', end=' ')
        try:
            option = int(input())
            if option < 1 or option > 3:
                raise OutsideOptionRange
            return option
        except ValueError:
            print('No se ingreso un numero.', end=' ')
        except OutsideOptionRange:
            print(f'La opcion {option} no existe.', end=' ')
        print('Intente de nuevo por favor.')

def get_retake_letter() -> str:
    letter_idx = -1
    while True:
        for key, value in LETTER_TO_LABEL_MAP.items():
            print(f'{key}: {value}', end=' ')
        print('\nIngrese el numero de la letra que quiere retomar:', end=' ')
        try:
            letter_idx = int(input())
            if letter_idx < 0 or letter_idx > 20:
                raise OutsideOptionRange
            return LABEL_TO_LETTER_MAP[letter_idx]
        except ValueError:
            print('No se ingreso un numero.', end=' ')
        except OutsideOptionRange:
            print(f'La letra {letter_idx} no existe.', end=' ')
        print('Intente de nuevo por favor.')

def create_dir_for_letter(dir_letter: str, u_letter='') -> tuple[bool, str]:
    directory_name = './' + dir_letter + '_' + u_letter + '/'
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
        return True, directory_name
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
        return True, directory_name
    except PermissionError:
        print(f"Permission denied: Unable to create '{directory_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return False, ''

def take_pictures(seconds: int, n: int, p_letter: str, u_letter: str) -> None:
    created_dir, directory = create_dir_for_letter(p_letter, u_letter)
    if not created_dir:
        return

    count = 0
    last_capture_time = time.time()
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Cannot read from camera.")
            break

        cv2.imshow(f'{n} pictures for letter {p_letter} each {seconds} seconds.', frame)

        if time.time() - last_capture_time >= seconds:
            filepath = directory + u_letter + p_letter + str(count) + '.jpg'
            cv2.imwrite(filepath, frame)
            print(f"Saved image {count + 1} at {filepath}")

            count += 1
            last_capture_time = time.time()

            if count == n:
                print(f"All {n} photos for letter {p_letter} taken.")
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User cancelled.")
            break

def take_more_photos_menu() -> bool:
    option = False
    while True:
        print('Si desea seguir tomando fotos ingrese 1, de lo contrario 0:', end=' ')
        try:
            option = int(input())
            if option < 0 or option > 1:
                raise OutsideOptionRange
            return True if option == 1 else False
        except ValueError:
            print('No se ingreso un numero.', end=' ')
        except OutsideOptionRange:
            print(f'La opcion {option} no existe.', end=' ')
        print('Intente de nuevo por favor.')

def start_letter_menu() -> int:
    letter_idx = -1
    while True:
        for key, value in LETTER_TO_LABEL_MAP.items():
            print(f'{key}: {value}', end=' ')
        print('\nIngrese el numero de la letra en la que quiere iniciar:', end=' ')
        try:
            letter_idx = int(input())
            if letter_idx < 0 or letter_idx > 20:
                raise OutsideOptionRange
            return letter_idx
        except ValueError:
            print('No se ingreso un numero.', end=' ')
        except OutsideOptionRange:
            print(f'La letra {letter_idx} no existe.', end=' ')
        print('Intente de nuevo por favor.')

def parse_args() -> tuple[int, int]:
    parser = argparse.ArgumentParser(
        description="Script that automatically takes pictures for each 21 classes."
    )
    parser.add_argument("--time", required=False, type=int, help='Enter time between photos.', default=3)
    parser.add_argument("--photos", required=False, type=int, help='Enter number of photos taken per letter.', default=40)
    args = parser.parse_args()

    return args.time, args.photos

if __name__ == '__main__':
    seconds_between_photos, total_photos_per_class = parse_args()

    while True:
        opt = first_menu()
        if opt == 3:
            print('Ha salido del programa.')
            quit()
        user_letter = get_user()
        print(user_letter)

        # Take all photos for all letters
        if opt == 1:
            start_letter_idx = start_letter_menu()
            for letter, idx in LETTER_TO_LABEL_MAP.items():
                if idx < start_letter_idx:
                    continue
                cam, w, h = open_cam()
                take_pictures(seconds_between_photos, total_photos_per_class, letter, user_letter)
                close_cam(cam)
                if letter == 'Y':
                    break
                if not take_more_photos_menu():
                    break
        # Take all photos for one letter
        else:
            cam, w, h = open_cam()
            retake_letter = get_retake_letter()
            take_pictures(seconds_between_photos, total_photos_per_class, retake_letter, user_letter)
            close_cam(cam)