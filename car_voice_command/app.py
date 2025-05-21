#Car Command V2 2023-06-12

import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BertTokenizer, BertForSequenceClassification
import soundfile as sf
import pyaudio
import wave
from tqdm.auto import tqdm
import os
import glob
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 4
LABELS = [
    "bukan command",
    "kunci pintu",
    "buka pintu",
    "nyala ac",
    "matiin ac",
    "turunin jendela",
    "naikin jendela",
    "buka bagasi",
    "tutup bagasi",
    "naikan suhu",
    "turunkan suhu",
    "turunkan volume",
    "naikan volume",
]

def record():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    print("Recording..")
    for i in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS))):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open('temp.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return b''.join(frames)

def speech_file_to_array_fn(speech_array=[], path=""):
    if path != "": speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, 16_000)
    wav = resampler(speech_array).squeeze().numpy()
    return wav

def predict_text(processor, model, speech=[], path=""):
    test_data = speech_file_to_array_fn(speech, path) if path != "" else speech
    inputs = processor(test_data, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)

def predict_class():
    raise NotImplementedError

if __name__ == '__main__':
    print("\033[93m"+"Initializing....."+"\033[0m")
    print("\033[92m"+"Downloading Wav2Vec2......."+"\033[0m")
    processor = Wav2Vec2Processor.from_pretrained("indonesian-nlp/wav2vec2-large-xlsr-indonesian")
    model = Wav2Vec2ForCTC.from_pretrained("indonesian-nlp/wav2vec2-large-xlsr-indonesian")

    print("\033[92m"+"Downloading Fine-tuned BERT........."+"\033[0m")
    if not 'bert_indo_car_commandv2.pth' in glob.glob("*"):
        os.system("gdown https://drive.google.com/uc?id=1-2HbYf_mpaDxJHTB_nnbv1j0EcdSZtqt")
    bert = BertForSequenceClassification.from_pretrained('cahya/bert-base-indonesian-522M', num_labels=13)
    tokenizer = BertTokenizer.from_pretrained('cahya/bert-base-indonesian-522M')
    bert.load_state_dict(torch.load('bert_indo_car_commandv2.pth', map_location=torch.device('cpu')))

    while True:
        print("")
        input("Press enter to record")
        audio_arr = record()

        text = predict_text(processor, model, path="temp.wav")[0]
        print("You said : ", text)
        if text == "":
            print("Command: ", LABELS[0])
        else:
            bert.eval()
            test_text_token = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                test_out = bert(**test_text_token)
            print("Command: ", LABELS[test_out.logits.softmax(1).argmax(1)])

