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

from opendr.perception.speech_transcription import (
    WhisperLearner,
    VoskLearner,
)


class AudioProcessor:
    def __init__(self, args):
        rospy.init_node('audio_processor')

        self.data_queue = Queue()

        # Initialize model
        self.backbone = args.backbone
        self.model_name = args.model_name
        self.model_path = args.model_path
        self.language = None if args.language.lower() == "none" else args.language 
        self.download_dir = args.download_dir

        if self.backbone == "whisper":
            if self.model_path is not None:
                name = self.model_path
            else:
                name = self.model_name
            # Load Whisper model
            self.audio_model = WhisperLearner(language=self.language)
            self.audio_model.load(name=name, download_dir=self.download_dir)
        else:
            # Load Vosk model
            self.audio_model = VoskLearner()
            self.audio_model.load(name=self.model_name, model_path=self.model_path, language=self.language, download_dir=self.download_dir)

        self.last_sample = b''

        self.temp_file = args.temp_file
        self.transcription = ['']

        self.sub = rospy.Subscriber(args.input_topic, AudioData, self.callback)

        # Start processing thread
        self.processing_thread = Thread(target=self.process_audio)
        self.processing_thread.start()

        self.sample_width = args.sample_width
        self.framerate = args.framerate

        self.n_sample = None

    def callback(self, data):
        # Add data to queue
        self.data_queue.put(data.data)

    def process_audio(self):
        while not rospy.is_shutdown():
            # Check if there is any data in the queue
            if not self.data_queue.empty():
                # Get the audio data from the queue
                # Concatenate our current audio data with the latest audio data.
                while not self.data_queue.empty():
                    data = self.data_queue.get()
                    self.last_sample += data

                # Write wav data to the temporary file.
                with wave.open(self.temp_file, 'wb') as f:
                    f.setnchannels(1)
                    f.setsampwidth(self.sample_width) 
                    f.setframerate(self.framerate)
                    # Convert audio data to numpy array
                    numpy_data = np.frombuffer(self.last_sample, dtype=np.int16)
                    if self.n_sample is not None:
                        numpy_data = numpy_data[(self.n_sample - 3200):]
                        self.n_sample = None

                    f.writeframes(numpy_data.tobytes())

                # Process audio
                if self.backbone == "vosk":
                    with open(self.temp_file, 'rb') as f:
                        while True:
                            data = f.read(4000)
                            if len(data) == 0:
                                break
                            # if self.recognizer.AcceptWaveform(data):
                            #     result = json.loads(self.recognizer.Result())
                            #     text = result['text'].strip()
                            # else:
                            #     result = json.loads(self.recognizer.PartialResult())
                            #     text = result['partial'].strip()
                            result = self.audio_model.infer(data)
                            text = text.data

                    # if text != '':
                    #     if phrase_complete:
                    #         self.transcription.append(text)
                    #     else:
                    #         self.transcription[-1] = text
                    if text == '' and self.transcription[-1] != '':
                        self.transcription.append('')
                    else:
                        self.transcription[-1] = text

                    # print(text)
                    os.system('cls' if os.name=='nt' else 'clear')
                    # print(len(self.transcription))
                    print(self.transcription)
                    for line in self.transcription:
                        print(line)

                else:
                    audio_array = whisper.load_audio(self.temp_file)
                    transcription_whisper = self.audio_model.infer(audio_array, builtin_transcribe=True)
                    segments = transcription_whisper.segments
                    if len(segments) > 1:
                        last_segment = segments[-1]
                        start_timestamp = last_segment['start']
                        self.n_sample = int(self.framerate * start_timestamp)

                        text = [segments[i]['text'].strip() for i in range(len(segments) - 1)]
                        text = " ".join(text)
                    else:   
                        text = transcription_whisper.data.strip()

                    os.system('cls' if os.name=='nt' else 'clear')
                    if self.n_sample is not None:
                        print(last_segment['text'])
                    else:
                        print(text)

                # Sleep to prevent busy looping
                # sleep(0.25)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--backbone', default='whisper', help='backbone to use for audio processing. Options: whisper, vosk', choices=['whisper', 'vosk'])
    parser.add_argument('--model-name', default='tiny', help='model to use for audio processing. Options: tiny, small, medium, large, en-us')
    parser.add_argument('--model-path', default=None, help='path to model')
    parser.add_argument('--download-dir', default=None, help='directory to download models to')
    parser.add_argument('--language', default="en", help='language to use for audio processing')
    parser.add_argument('--temp-file', default='./temp.wav', help='temporary file foraudio data')
    parser.add_argument('--input-topic', default='/audio/audio', help='name of the topic to subscribe')
    parser.add_argument('--output-topic', default='/audio/transcription', help='name of the topic to publish')
    parser.add_argument('--sample-width', type=int, default=2, help='sample width for audio data')
    parser.add_argument('--framerate', type=int, default=16000, help='framerate for audio data')
    args = parser.parse_args()

    try:
        node = AudioProcessor(args)
        node.spin()
    except rospy.ROSInterruptException:
        pass

