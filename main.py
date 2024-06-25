import numpy as np
from PIL import ImageGrab
import cv2
import time
from pynput.mouse import Button, Controller
from pynput import keyboard
import requests
from datetime import datetime
from dotenv import dotenv_values
from re import sub
from json import JSONDecodeError, loads as loadjson
import threading


def get_env_value(key, message, allow_empty=False):
    try:
        value = dotenv_values(".env")[key]
        if not value and not allow_empty:
            raise KeyError(f"Empty value found for {key}")
        return value
    except KeyError:
        print(f"Error: {message}")
        exit()


# we don't care if DEBUG is set
debug = dotenv_values(".env").get("DEBUG", "false").lower() == "true"

hotkey_enabled = get_env_value(
    "HOTKEY_ENABLED", ".env is missing or HOTKEY_ENABLED has not been set. Refer to .env_example on how to add the variable.").lower() == "true"

hotkey = get_env_value(
    "HOTKEY", ".env is missing or HOTKEY has not been defined. Refer to .env_example on how to define the variable.") if hotkey_enabled else None

notify_enabled = get_env_value(
    "NOTIFY_ENABLED", ".env is missing or NOTIFY_ENABLED has not been set. Refer to .env_example on how to add the variable.").lower() == "true"


notify_settings = {
    "url":  get_env_value(
        "NOTIFY_API_URL", ".env is missing or NOTIFY_API_URL has not been defined. Refer to .env_example on how to add the API url.", False),
    "api_key": get_env_value(
        "NOTIFY_API_KEY", ".env is missing or NOTIFY_API_KEY has not been defined. Refer to .env_example on how to set the API settings.", True),
    "pm_am": get_env_value(
        "NOTIFY_12H_TIME", ".env is missing or NOTIFY_12H_TIME has not been defined. Refer to .env_example on how to set the API settings.", False).lower() == "true"
} if notify_enabled else None

try:
    colors = loadjson(get_env_value(
        "COLORS", "COLORS has not been set. Refer to .env_example on how to set the colors needed to recognize the accept button."))
except JSONDecodeError:
    print(f"Make sure the colors are formatted correctly!")
    exit()


try:
    percent_color_threshold = float(get_env_value("PERCENT_COLOR_THRESHOLD",
                                                  "PERCENT_COLOR_THRESHOLD has not been set. Refer to .env_example on how to set the accept button parameters."))
    aspect_ratio_threshold = float(get_env_value("ASPECT_RATIO_THRESHOLD",
                                                 "ASPECT_RATIO_THRESHOLD has not been set. Refer to .env_example on how to set the accept button parameters."))

except ValueError:
    print(f"percent_color_threshold or ASPECT_RATIO_THRESHOLD need to be numbers!!")
    exit()


try:
    bbox_left = int(get_env_value("BOUNDING_BOX_LEFT",
                    "BOUNDING_BOX_LEFT has not been set. Refer to .env_example on how to set the bounding box parameters."))
    bbox_upper = int(get_env_value("BOUNDING_BOX_UPPER",
                                   "BOUNDING_BOX_UPPER has not been set. Refer to .env_example on how to set the bounding box parameters."))
    bbox_right = int(get_env_value("BOUNDING_BOX_RIGHT",
                                   "BOUNDING_BOX_RIGHT has not been set. Refer to .env_example on how to set the bounding box parameters."))
    bbox_lower = int(get_env_value("BOUNDING_BOX_LOWER",
                                   "BOUNDING_BOX_LOWER has not been set. Refer to .env_example on how to set the bounding box parameters."))

except ValueError:
    print(f"BOUNDING_BOX parameters need to be numbers!!")
    exit()


def click(position, mouse_controller):
    # position: a tuple (x,y) coordinates
    # # Move pointer to button and click
    mouse_controller.position = position
    mouse_controller.press(Button.left)
    mouse_controller.release(Button.left)


def notify(url, api_key="", pm_am=False):
    """
    Sends a request to the Notify service to send a notification regarding the match
    url: the url of the service
    api_key: the api_key needed to access (default to an empty string)
    pm_am: a bool whether or not 12h time is enabled (default to False)

    """
    current_time = None
    if pm_am:
        current_time = datetime.now().strftime("%I:%M:%S %p")
        # gets rid of the leading zero on the hour
        current_time = sub(r"0(\d)", r"\1", current_time)
    else:
        current_time = datetime.now().strftime("%H:%M:%S")
    # current_time = sub(datetime.now().strftime("%I:%M:%S %p") if pm_am datetime.now().strftime("%I:%M:%S %p".replace() if pm_am else "%H:%M:%S")
    requests.request("POST", url, json={
        "title": "CS2 Accepter: Match accepted",
        "message": f"It is time to start playing! {current_time}",
        "tags": ["Urgent", "CS2 Accepter"]
    }, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else None
    })


def check_for_color(img, target_color_hex):
    # Convert the target color from hex to BGR
    target_color_bgr = tuple(
        int(target_color_hex[i:i+2], 16) for i in (4, 2, 0))

    # Check if the target color exists in the image
    target_color_np = np.array(target_color_bgr, dtype=np.uint8)
    mask = cv2.inRange(img, target_color_np, target_color_np)
    color_exists = np.any(mask)

    # Calculate the total number of pixels in the image
    total_pixels = img.shape[0] * img.shape[1]
    # Count the number of pixels matching the target color
    color_count = np.count_nonzero(mask)

    # Calculate the percentage of pixels that are the target color
    percentage = (color_count / total_pixels) * \
        100 if total_pixels > 0 else 0.0

    return color_exists, percentage


class AutoAccepter:
    accepted = False

    # notifier can either not be set or be a callback to a function
    def __init__(self, bbox, colors, percent_color_threshold=40, aspect_ratio_threshold=2.0, notify_enabled=False, notify_settings={"url": "", "api_key": "", "pm_am": False}, debug=False):
        self.bbox = bbox
        self.colors = colors
        self.percent_color_threshold = percent_color_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.debug = debug
        self.notify = notify_enabled
        self.notify_settings = notify_settings
        self.mouse_controller = Controller()

    # returns whether or not a color can be found, and finds the one most common
    def check_for_autoaccept_colors(self, img):
        color_found = 0
        for color in self.colors:
            exists, percentage = check_for_color(img, color)
            if (exists and percentage > color_found):
                color_found = percentage
        return color_found > 0.00, color_found

    def find_accept_button(self, img, find_best=True, accuracy=0.01):
        """
        returns coordinates if found

        find_best: returns the match with the highest color %, otherwise it just returns the last find

        accuracy: the constant that gets multiplied with the approx contour perimeter. Default is 0.01
        """
        max_percent_color = 0
        button_middle_x = 0
        button_middle_y = 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detection
        edged = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over detected contours
        for contour in contours:
            # Approximate the contour with accuracy proportional to the contour perimeter
            epsilon = accuracy * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # If the approximated contour has 4 vertices, it's a rectangle, otherwise we don't care
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            #    Calculate aspect ratio
            aspect_ratio = float(w) / h
            rectangle_image = img[y:y+h, x:x+w]

            contains_colour, percent_color = self.check_for_autoaccept_colors(
                rectangle_image)

            if (not contains_colour or percent_color < self.percent_color_threshold or aspect_ratio < self.aspect_ratio_threshold):
                continue

            if (not find_best or percent_color > max_percent_color):
                max_percent_color = percent_color
                button_middle_x = (
                    bbox_left + x + bbox_left + x + w) / 2
                button_middle_y = (
                    bbox_upper + y + bbox_upper + y + h) / 2

            if self.debug:
                cv2.rectangle(
                    img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        return max_percent_color != 0, (button_middle_x, button_middle_y)

    def start(self):
        t = threading.current_thread()
        # for debug
        last_time = time.time()
        while getattr(t, "do_run", True):
            if self.debug:
                print('loop took {} seconds'.format(time.time()-last_time))
                last_time = time.time()

            # grab and parse screen
            printscreen = np.array(ImageGrab.grab(
                bbox=self.bbox))
            img = cv2.cvtColor(printscreen, cv2.COLOR_BGRA2RGB)

            button_color_present, _ = self.check_for_autoaccept_colors(img)

            if self.debug:
                cv2.imshow('window', img)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

            if (button_color_present and self.accepted):
                continue

            if (not button_color_present and not self.accepted):
                continue

            if (not button_color_present and self.accepted):
                self.accepted = False
                continue

            button_found, button_position = self.find_accept_button(img)
            if not button_found:
                continue

            time.sleep(1)
            click(button_position, self.mouse_controller)

            if self.notify:
                notify(self.notify_settings["url"],
                       self.notify_settings["api_key"], notify_settings["pm_am"])

            self.accepted = True

            if self.debug:
                cv2.imshow('window', img)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
                # wait 3 second to allow for recognition the red box around the button, debug purposes only
                time.sleep(3)

        if (self.debug):
            cv2.destroyAllWindows()
        self.accepted = False


def cleanupKeyCode(keycode):
    """
    Converts keycode like "<ctrl>+<alt>+<shift>+h" into something readable, like "Ctrl+Alt+Shift+H"
    """
    # take out the less than and greater than signs, <, >
    keycode = sub(r"<([^>]*)>", r"\1", hotkey)
    # split by +, so we can capitalize the first char
    keycode = keycode.split("+")
    keycode = map(lambda x: x.capitalize(), keycode)

    return "+".join(keycode)


class HotkeyManager:
    def __init__(self, hotkey, start_class):
        self.activate_thread = False
        self.hotkey = keyboard.HotKey(
            keyboard.HotKey.parse(hotkey),
            self.on_activate
        )
        self.listener = keyboard.Listener(
            on_press=self.for_canonical(self.hotkey.press),
            on_release=self.for_canonical(self.hotkey.release)
        )
        self.start_class = start_class
        self.hotkey = hotkey

    def for_canonical(self, f):
        return lambda k: f(self.listener.canonical(k))

    def on_activate(self):
        if self.activate_thread:
            print("Auto Accepter stopped!")
            self.activate_thread.do_run = False
            self.activate_thread = False
        else:
            print("Auto Accepter started!")
            self.activate_thread = threading.Thread(
                target=self.start_class.start)
            self.activate_thread.start()

    def start(self):
        self.listener.start()
        try:
            print(
                f"Press Ctrl+C to exit, or {cleanupKeyCode(self.hotkey)} to trigger the hotkey.")
            while self.listener.is_alive():
                self.listener.join(0.5)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Exiting...")
            exit()


bbox = (bbox_left, bbox_upper, bbox_right, bbox_lower)
auto_accepter = AutoAccepter(
    bbox, colors, percent_color_threshold, aspect_ratio_threshold, notify_enabled, notify_settings, debug)

if (hotkey):
    hotkey_manager = HotkeyManager(hotkey, auto_accepter)
    hotkey_manager.start()
else:
    print(f"Press Ctrl+C to exit.")
    auto_accepter.start()
