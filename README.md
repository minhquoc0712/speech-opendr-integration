# speech-opendr-integration

Launch audio capture node:
```
roslaunch audio_capture capture_wave.launch
```

Here is some example of run transcription node with some parameters:

Whisper: model name, no download dir, download to cache.
```
rosrun opendr_perception speech_transcription_node.py --backbone whisper --model-name tiny.en
```

Whisper: model name, with download dir
```
rosrun opendr_perception speech_transcription_node.py --backbone whisper --model-name tiny.en --download-dir "./whisper_model/"
```

Whisper: model path, assuming you have downloaded the model checkpoint.
```
rosrun opendr_perception speech_transcription_node.py --backbone whisper --model-path "./whisper_model/tiny.en.pt"
```

Vosk: Model name, no download dir
```
rosrun opendr_perception speech_transcription_node.py --backbone vosk --model-name "vosk-model-en-us-0.22-lgraph"
```

Vosk: Model path, assuming you have downloaded the model checkpoint.
```
rosrun opendr_perception speech_transcription_node.py --backbone vosk --model-path "./vosk-model-en-us-0.22-lgraph"
```

Vosk: language.
```
rosrun opendr_perception speech_transcription_node.py --backbone vosk --language en-us
```
