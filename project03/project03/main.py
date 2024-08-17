import threading
import speech_recognition as sr
from sense_hat import SenseHat
import curls
import squats
import google.generativeai as genai

# Initialize the recognizer, Sense HAT, and Gemini API
recognizer = sr.Recognizer()
sense = SenseHat()

# Configure the Gemini API
api_key = 'AIzaSyDK3HYgFK6ndeE6FnP6siP1lYkRmmnHbzM'
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

# Global variables to track the current mode and threads
current_mode = "curls"  # Default mode
curl_thread = None
squat_thread = None

# Function to capture speech and process with Gemini API
def capture_speech():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio_input = recognizer.listen(source)
        try:
            recognized_text = recognizer.recognize_google(audio_input)
            print("You Said:", recognized_text)
            return recognized_text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not get results from Google Speech Recognition service; {e}")
            return None

def process_with_gemini(text):
    try:
        response = model.generate_content(text)
        response_text = response.text
        print("API Response:", response_text)
        return response_text
    except Exception as e:
        print(f"An error occurred while contacting Gemini API: {e}")
        return None

# Function to switch workout modes
def switch_mode(new_mode):
    global current_mode, curl_thread, squat_thread
    if new_mode == current_mode:
        return
    
    print(f"Switching to {new_mode} mode")
    sense.clear()
    
    if current_mode == "curls":
        if curl_thread:
            curl_thread.do_run = False
            curl_thread.join()
    elif current_mode == "squats":
        if squat_thread:
            squat_thread.do_run = False
            squat_thread.join()
    
    if new_mode == "curls":
        curl_thread = threading.Thread(target=curls.start_curls)
        curl_thread.start()
        sense.show_message("Curls Mode", text_colour=[255, 255, 255], back_colour=[0, 0, 0])
    elif new_mode == "squats":
        squat_thread = threading.Thread(target=squats.start_squats)
        squat_thread.start()
        sense.show_message("Squats Mode", text_colour=[255, 255, 255], back_colour=[0, 0, 0])
    
    current_mode = new_mode

# Main loop
if _name_ == "_main_":
    # Start with the default mode
    curl_thread = threading.Thread(target=curls.start_curls)
    curl_thread.start()

    while True:
        recognized_text = capture_speech()
        if recognized_text:
            response_text = process_with_gemini(recognized_text)
            if response_text:
                if "squats" in response_text.lower():
                    switch_mode("squats")
                elif "curls" in response_text.lower():
                    switch_mode("curls")