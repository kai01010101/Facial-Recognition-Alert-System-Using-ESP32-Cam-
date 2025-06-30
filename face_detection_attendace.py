import cv2
import urllib.request
import numpy as np
import os
import requests
import threading
import time
import hashlib
from datetime import datetime
import face_recognition
import signal
import sys
import dlib

# === CONFIG ===
image_folder   = r'images'                       # Known faces folder
camera_url     = 'http://192.168.1.75/jpg'      # ESP32-CAM snapshot URL
unknown_folder = "unknown_faces"
alerts_enabled = threading.Event()               # Use event for thread-safe toggle
alerts_enabled.set()                             # Alerts enabled by default

# === TELEGRAM BOT CONFIG ===
TELEGRAM_BOT_TOKEN = "8124309656:AAGLidGsVHZHnqNA0c5UCNDDQK_GkWoXNJc"
TELEGRAM_CHAT_ID   = "6209678025"
TELEGRAM_API_URL   = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# === SETUP ===
os.makedirs(unknown_folder, exist_ok=True)

# Load known faces
images, classNames = [], []
for file in os.listdir(image_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, file)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            classNames.append(os.path.splitext(file)[0])
        else:
            print(f"⚠️ Warning: Failed to load image {img_path}")

def findEncodings(img_list):
    enc_list = []
    for img in img_list:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(rgb)
        if enc:
            enc_list.append(enc[0])
        else:
            print("⚠️ Warning: No face found in one of the known images.")
    return enc_list

encodeListKnown = findEncodings(images)
print(f"🔐 Loaded {len(encodeListKnown)} known faces.")

def _raw_face_locations(img, number_of_times_to_upsample=1, model="hog"):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of dlib 'rect' objects of found face locations
    """
    if model == "cnn":
        return cnn_face_detector(img, number_of_times_to_upsample)
    else:
        return face_detector(img, number_of_times_to_upsample)

def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)
# === TELEGRAM HELPERS ===
def tg_send_message(text):
    try:
        response = requests.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
        time.sleep(0.3)  # Slight delay to avoid flooding Telegram API
        if not response.ok:
            print(f"⚠️ Telegram message error: {response.text}")
    except Exception as e:
        print(f"⚠️ Telegram message exception: {e}")

def tg_send_photo(photo_path, caption=""):
    try:
        with open(photo_path, 'rb') as photo:
            response = requests.post(
                f"{TELEGRAM_API_URL}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                files={"photo": photo},
                timeout=15
            )
            time.sleep(0.5)
            if not response.ok:
                print(f"⚠️ Telegram photo error: {response.text}")
    except Exception as e:
        print(f"⚠️ Telegram photo exception: {e}")

# === ATTENDANCE TRACKING ===
seen_names = set()
seen_hashes = set()
lock = threading.Lock()

def notify_attendance(name, crop=None):
    if not alerts_enabled.is_set():
        return

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if name == "Unknown":
        caption = f"🚨 *Unknown person detected!*  \n🕒 `{ts}`"
        if crop is not None:
            crop = np.ascontiguousarray(crop)
            crop_hash = hashlib.md5(crop.tobytes()).hexdigest()
            with lock:
                if crop_hash not in seen_hashes:
                    seen_hashes.add(crop_hash)
                    filename = f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    path = os.path.join(unknown_folder, filename)
                    cv2.imwrite(path, crop)
                    # Send alert in a separate thread to not block main loop
                    threading.Thread(target=tg_send_photo, args=(path, caption), daemon=True).start()
    else:
        with lock:
            if name not in seen_names:
                seen_names.add(name)
                message = f"✅ *{name} marked present!*  \n🕒 `{ts}`"
                threading.Thread(target=tg_send_message, args=(message,), daemon=True).start()

def send_last_unknowns(count=5):
    files = sorted(os.listdir(unknown_folder), reverse=True)[:count]
    for f in files:
        threading.Thread(target=tg_send_photo, args=(os.path.join(unknown_folder, f),), daemon=True).start()

# === TELEGRAM BOT THREAD ===
def telegram_bot():
    offset = None
    print("🤖 Telegram bot running...")

    while True:
        try:
            params = {"timeout": 10}
            if offset:
                params["offset"] = offset
            r = requests.get(f"{TELEGRAM_API_URL}/getUpdates", params=params, timeout=15)
            data = r.json()

            if not data.get("ok"):
                print("⚠️ Telegram API returned not ok")
                time.sleep(5)
                continue

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message", {})
                chat_id = str(message.get("chat", {}).get("id"))
                text = message.get("text", "").strip()

                if chat_id != TELEGRAM_CHAT_ID:
                    threading.Thread(target=tg_send_message, args=("🚫 Unauthorized access.",), daemon=True).start()
                    continue

                # Command handling
                if text == "/start":
                    threading.Thread(target=tg_send_message, args=("👋 Welcome to *AI Attendance Bot*!\nType /help for commands.",), daemon=True).start()
                elif text == "/help":
                    help_msg = (
                        "📋 *Commands:*\n"
                        "`/photo` – Capture live image\n"
                        "`/status` – System status\n"
                        "`/unknowns` – Show last unknowns\n"
                        "`/knowns` – Show known faces\n"
                        "`/reset` – Clear attendance memory\n"
                        "`/stop_alerts` – Disable alerts\n"
                        "`/start_alerts` – Enable alerts"
                    )
                    threading.Thread(target=tg_send_message, args=(help_msg,), daemon=True).start()
                elif text == "/photo":
                    try:
                        resp = urllib.request.urlopen(camera_url, timeout=5)
                        img = cv2.imdecode(np.frombuffer(resp.read(), np.uint8), -1)
                        cv2.imwrite("temp.jpg", img)
                        threading.Thread(target=tg_send_photo, args=("temp.jpg", "📸 Live Snapshot"), daemon=True).start()
                    except Exception:
                        threading.Thread(target=tg_send_message, args=("⚠️ Could not fetch camera image.",), daemon=True).start()
                elif text == "/status":
                    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ip = camera_url.split('/')[2]
                    status_msg = (
                        f"📡 *System Status:*\n"
                        f"`Time:` {now}\n"
                        f"`Camera:` {ip}\n"
                        f"`Known:` {len(classNames)} faces\n"
                        f"`Alerts:` {'Enabled' if alerts_enabled.is_set() else 'Disabled'}"
                    )
                    threading.Thread(target=tg_send_message, args=(status_msg,), daemon=True).start()
                elif text == "/unknowns":
                    send_last_unknowns()
                elif text == "/knowns":
                    knowns_msg = "🧠 *Known Users:*\n" + "\n".join(f"• `{name}`" for name in classNames)
                    threading.Thread(target=tg_send_message, args=(knowns_msg,), daemon=True).start()
                elif text == "/reset":
                    with lock:
                        seen_names.clear()
                        seen_hashes.clear()
                    threading.Thread(target=tg_send_message, args=("🔄 Attendance memory cleared.",), daemon=True).start()
                elif text == "/stop_alerts":
                    alerts_enabled.clear()
                    threading.Thread(target=tg_send_message, args=("🔕 Alerts disabled.",), daemon=True).start()
                elif text == "/start_alerts":
                    alerts_enabled.set()
                    threading.Thread(target=tg_send_message, args=("🔔 Alerts enabled.",), daemon=True).start()
                else:
                    threading.Thread(target=tg_send_message, args=("❓ Unknown command. Type /help.",), daemon=True).start()

        except Exception as e:
            print(f"⚠️ Telegram bot error: {e}")
            time.sleep(5)

# === GRACEFUL SHUTDOWN HANDLER ===
def signal_handler(sig, frame):
    print("\n🛑 Exiting gracefully...")
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === START BOT THREAD ===
threading.Thread(target=telegram_bot, daemon=True).start()

# === MAIN CAMERA LOOP ===
frame_skip = 2   # Process every 2nd frame to reduce CPU load
frame_count = 0

while True:
    frame_count += 1
    try:
        resp = urllib.request.urlopen(camera_url, timeout=5)
        frame = cv2.imdecode(np.frombuffer(resp.read(), np.uint8), -1)
    except Exception as e:
        print(f"❌ Camera error: {e}")
        time.sleep(1)
        continue

    if frame_count % frame_skip != 0:
        # Skip processing for smoother performance
        cv2.imshow('🛡️ ESP32-CAM Feed (Press Q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

face_detector = dlib.get_frontal_face_detector()
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])

face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
def face_encode(face_image, known_face_locations=None, num_jitters=1, model="small"):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

def _raw_face_locations(img, number_of_times_to_upsample=1, model="hog"):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of dlib 'rect' objects of found face locations
    """
    if model == "cnn":
        return cnn_face_detector(img, number_of_times_to_upsample)
    else:
        return face_detector(img, number_of_times_to_upsample)
def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    pose_predictor = pose_predictor_68_point

    if model == "small":
        pose_predictor = pose_predictor_5_point

    return [pose_predictor(face_image, face_location) for face_location in face_locations]

def face_locations(img, number_of_times_to_upsample=1, model="hog"):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """
    if model == "cnn":
        return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, "cnn")]
    else:
        return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, model)]
    
    
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    faces = face_locations(rgb)
    encs = face_encode(rgb, faces)


def face_dist(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

def comp_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


for encodeFace, (top, right, bottom, left) in zip(encs, faces):
    matches = comp_faces(encodeListKnown, encodeFace)
    faceDis = face_dist(encodeListKnown, encodeFace)

    y1, x2, y2, x1 = top * 4, right * 4, bottom * 4, left * 4
    crop = frame[y1:y2, x1:x2]

    if not matches or not any(matches):
            notify_attendance("Unknown", crop)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    else:
            idx = np.argmin(faceDis)
            name = classNames[idx].upper()
            notify_attendance(name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('🛡️ ESP32-CAM Feed (Press Q to quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
