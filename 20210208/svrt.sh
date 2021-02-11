#!/bin/bash
./master/master &\
sleep 3;
python3 deepSpeech/streaming_stt.py -m ./deepSpeech/deepspeech-0.9.3-models.tflite -s ./deepSpeech/deepspeech-0.9.3-models.scorer -d 7 &\
sleep 5;
./matrix/mo & ./odas/bin/odaslive -vc ./matrix/matrix_voice.cfg

