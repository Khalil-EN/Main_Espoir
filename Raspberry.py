import pytesseract
import cv2
import speech_recognition as sr
import pyttsx3
import os
import numpy as np
import pyaudio
import wave
from pydub import AudioSegment
from openai import OpenAI
from gtts import gTTS
import pygame
from googleapiclient.discovery import build
from google.oauth2 import service_account
import io
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaFileUpload
import time




SCOPES=['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE="service_account.json"
PARENT_FOLDER_ID="folder_id_from_google_drive(after /folders/)"

def authentificate():
    creds=service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return creds

def authenticate():
    # Load service account credentials from JSON file
    creds = service_account.Credentials.from_service_account_file('service_account.json')
    # Create a service object for interacting with the Google Drive API
    service = build('drive', 'v3', credentials=creds)
    return service

def authenticate2():
    creds = service_account.Credentials.from_service_account_file('service_account.json', scopes=['https://www.googleapis.com/auth/drive'])
    service = build('drive', 'v3', credentials=creds)
    return service


def download_photo(file_id, dest_path):
    creds = authentificate()
    service = build('drive', 'v3', credentials=creds)
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(dest_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))


def get_file_id(file_name, parent_folder_id):
    creds = authentificate()
    service = build('drive', 'v3', credentials=creds)
    query = f"name='{file_name}' and '{parent_folder_id}' in parents"
    response = service.files().list(q=query, fields='files(id)').execute()
    files = response.get('files', [])
    if files:
        return files[0]['id']
    else:
        print(f"File '{file_name}' not found in folder with ID '{parent_folder_id}'")
        return None



def update_photo(file_id, new_photo_path):
    service = authenticate2()
    media = MediaFileUpload(new_photo_path)
    service.files().update(fileId=file_id, media_body=media).execute()
    print(f"Photo with ID {file_id} updated successfully.")


# Initialize webcam
webcam = cv2.VideoCapture(0)



client = OpenAI(api_key = 'openai_key')



def get_transcription_from_whisper():


    # Set the audio parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 2048
    SILENCE_THRESHOLD = 300  # Silence threshold
    SPEECH_END_TIME = 1.0  # Time of silence to mark the end of speech

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...Waiting for speech to begin.")

    frames = []
    silence_frames = 0
    is_speaking = False
    total_frames = 0

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        total_frames += 1

        # Convert audio chunks to integers
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Check if user has started speaking
        if np.abs(audio_data).mean() > SILENCE_THRESHOLD:
            is_speaking = True

        # Detect if the audio chunk is silence
        if is_speaking:
            if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
                silence_frames += 1
            else:
                silence_frames = 0

        # End of speech detected
        if is_speaking and silence_frames > SPEECH_END_TIME * (RATE / CHUNK):
            print("End of speech detected.")
            break

    # Stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    print("Finished recording.")
    combined_audio_data = b''.join(frames)

    # Convert raw data to an AudioSegment object
    audio_segment = AudioSegment(
        data=combined_audio_data,
        sample_width=audio.get_sample_size(FORMAT),
        frame_rate=RATE,
        channels=CHANNELS
    )

    # Export as a compressed MP3 file with a specific bitrate
    audio_segment.export("output_audio_file.mp3", format="mp3", bitrate="32k")

    audio_file = open("output_audio_file.mp3", "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
    )
    # Return the transcript text
    return transcript.text





while True:
    try:
        # Capture frame from webcam
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)

        # Recognize speech
        words = get_transcription_from_whisper()
        print("Recognized:", words)

        # Check for wake word or key press
        if "hi" in words.lower():
            cv2.imwrite(filename='saved_img.jpg', img=frame) # replace saved_img.jpg with the appropriate filename
            print("Image saved!")
            file_id = get_file_id('lamp.png', PARENT_FOLDER_ID) # replace lamp.png with the appropriate filename
            update_photo(file_id,"saved_img.jpg") # replace saved_img.jpg with the appropriate filename
            time.sleep(15)
            file_id1=get_file_id("captions.txt",PARENT_FOLDER_ID ) # replace caption.txt with the appropriate filename
            download_photo(file_id1, 'captions.txt') # replace captions.txt with the appropriate filename
            with open('captions.txt', 'r') as f:
                text = f.read()
                if string is None or string.strip() == "":
                    print("Error: No text to speak.")
                else:

                    tts_english = gTTS(text=string, lang='en')


                    tts_english.save("output_english.mp3")

# Play the speech using the default audio player
                if os.path.exists("output_english.mp3"):
    # Initialize the Pygame mixer
                    pygame.mixer.init()

    # Load the output file
                    pygame.mixer.music.load("output_english.mp3")

    # Play the loaded file
                    pygame.mixer.music.play()

    # Wait for the playback to finish
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
                else:
                    print("Output file not found.")

        elif "قراءة" in words.lower()  or cv2.waitKey(1) & 0xFF == ord('z'):
            # Save captured image
            cv2.imwrite(filename='saved_img.jpg', img=frame) # replace saved_img.jpg with the appropriate filename
            print("Image saved!")

            text = "هل أنت في حاجة إلى اللغة العربية أم لغة أجنبية؟"
            tts_arabic2=gTTS(text=text, lang='ar')
            tts_arabic2.save("output_arabic2.mp3")
            if os.path.exists("output_arabic2.mp3"):

                pygame.mixer.init()

                pygame.mixer.music.load("output_arabic2.mp3")
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
            else:
                print("Output file not found.")

            response=get_transcription_from_whisper()
            print("Recognized:", response)
            if response=="العربية":
                # Perform OCR on the saved image
                img = cv2.imread('saved_img.jpg')
                string = pytesseract.image_to_string(img, lang='ara')
                print("OCR Result:", string)
                if string is None or string.strip() == "":
                    print("Error: No text to speak.")
                else:

                    tts_english = gTTS(text=string, lang='ar')


                    tts_english.save("output_arabic.mp3")
                if os.path.exists("output_arabic.mp3"):
    # Initialize the Pygame mixer
                    pygame.mixer.init()

    # Load the output file
                    pygame.mixer.music.load("output_arabic.mp3")

    # Play the loaded file
                    pygame.mixer.music.play()

    # Wait for the playback to finish
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
                else:
                    print("Output file not found.")
            else:
                # Perform OCR on the saved image
                img = cv2.imread('saved_img.jpg')
                string = pytesseract.image_to_string(img)
                print("OCR Result:", string)
                if string is None or string.strip() == "":
                    print("Error: No text to speak.")
                else:

                    tts_english = gTTS(text=string, lang='en')


                    tts_english.save("output_english.mp3")

# Play the speech using the default audio player
                if os.path.exists("output_english.mp3"):
    # Initialize the Pygame mixer
                    pygame.mixer.init()

    # Load the output file
                    pygame.mixer.music.load("output_english.mp3")

    # Play the loaded file
                    pygame.mixer.music.play()

    # Wait for the playback to finish
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)  # Adjust the tick rate as needed
                else:
                    print("Output file not found.")

            #break

    except sr.UnknownValueError:
        print("Speech recognition could not understand audio.")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    except cv2.error as e:
        print("Error capturing image from webcam:", e)
    except Exception as e:
        print("An error occurred:", e)
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()