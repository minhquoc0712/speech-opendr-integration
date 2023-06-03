#! /usr/bin/env python 

import rospy
from audio_common_msgs.msg import AudioData
from datetime import datetime, timedelta
from queue import Queue
from threading import Thread
from time import sleep
from tempfile import NamedTemporaryFile
import numpy as np
import io
import wave
import os
import torch
import whisper
import time
import argparse
import json
from vosk import Model, KaldiRecognizer


class AudioProcessor:
    def __init__(self, args):
        rospy.init_node('audio_processor')

        self.data_queue = Queue()

        # Initialize model
        self.model = args.model
        self.non_english = args.non_english
        if self.model == "vosk":
            # Load Vosk model
            self.vosk_model = Model(lang="en-us")
            self.recognizer = KaldiRecognizer(self.vosk_model, 16000)
        elif self.model != "large" and not self.non_english:
            self.model = self.model + ".en"
            self.audio_model = whisper.load_model(self.model)

        self.phrase_timeout = args.phrase_timeout
        self.phrase_time = None
        self.last_sample = b''

        self.temp_file = args.temp_file
        self.transcription = ['']

        self.sub = rospy.Subscriber(args.input_topic, AudioData, self.callback)

        # Start processing thread
        self.processing_thread = Thread(target=self.process_audio)
        self.processing_thread.start()

        self.sample_width = args.sample_width
        self.framerate = args.framerate

    def callback(self, data):
        now = rospy.get_time()
        # Add data to queue
        self.data_queue.put((now, data.data))

    def process_audio(self):
        while not rospy.is_shutdown():
            now = rospy.get_time()
            # Check if there is any data in the queue
            if not self.data_queue.empty():
                # Get the audio data from the queue

                # print(now, self.phrase_time, self.phrase_timeout)
                phrase_complete = False
                if self.phrase_time and now - self.phrase_time > self.phrase_timeout:
                    self.last_sample = b''
                    phrase_complete = True

                # print(phrase_complete)
                self.phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not self.data_queue.empty():
                    now, data = self.data_queue.get()
                    self.last_sample += data

                # Write wav data to the temporary file.
                with wave.open(self.temp_file, 'wb') as f:
                    f.setnchannels(1)
                    f.setsampwidth(self.sample_width) 
                    f.setframerate(self.framerate)
                    # Convert audio data to numpy array
                    numpy_data = np.frombuffer(self.last_sample, dtype=np.int16)
                    f.writeframes(numpy_data.tobytes())

                # Process audio
                if self.model == "vosk":
                    with open(self.temp_file, 'rb') as f:
                        while True:
                            data = f.read(4000)
                            if len(data) == 0:
                                break
                            if self.recognizer.AcceptWaveform(data):
                                result = json.loads(self.recognizer.Result())
                                text = result['text'].strip()
                            else:
                                result = json.loads(self.recognizer.PartialResult())
                                text = result['partial'].strip()

                    if phrase_complete:
                        self.transcription.append(text)
                    else:
                        self.transcription[-1] = text

                    os.system('cls' if os.name=='nt' else 'clear')
                    print(len(self.transcription))
                    for line in self.transcription:
                        print(line)

                else:
                    result = self.audio_model.transcribe(self.temp_file, fp16=torch.cuda.is_available())
                    text = result['text'].strip()
                    # no_speech_probs = result['no_speech_probs']
                    # for segment in result['segments']:
                    #     print(segment['text'])

                    if phrase_complete:
                        self.transcription.append(text)
                    else:
                        self.transcription[-1] = text

                    # result_segments = [segment['text'] for segment in result['segments']]
                    # if phrase_complete:
                    #     self.transcription.append(result_segments)
                    # else:
                    #     self.transcription[-1] = result_segments

                    os.system('cls' if os.name=='nt' else 'clear')
                    for line in self.transcription:
                        print(line)
                    # for segments in self.transcription:
                    #     for line in segments:
                    #         print(line)
                    #     print()


                # Sleep to prevent busy looping
                # sleep(0.25)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', default='tiny', help='model to use for audio processing. Options: tiny, large, vosk')
    parser.add_argument('--non_english', action='store_true', help='set if model is non-english')
    parser.add_argument('--phrase_timeout', type=float, default=1.0, help='timeout for phrases')
    parser.add_argument('--temp_file', default='./temp.wav', help='temporary file foraudio data')
    parser.add_argument('--input_topic', default='/audio/audio', help='name of the topic to subscribe')
    parser.add_argument('--output_topic', default='/audio/transcription', help='name of the topic to publish')
    parser.add_argument('--sample_width', type=int, default=2, help='sample width for audio data')
    parser.add_argument('--framerate', type=int, default=16000, help='framerate for audio data')
    args = parser.parse_args()

    try:
        node = AudioProcessor(args)
        node.spin()
    except rospy.ROSInterruptException:
        pass

