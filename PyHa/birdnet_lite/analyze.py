import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import tflite_runtime.interpreter as tflite
except:
    from tensorflow import lite as tflite

import argparse
import operator
import librosa
import numpy as np
import math
import time
import pandas as pd
from sys import exit
import logging
from pathlib import Path

def loadModel():

    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX
    global MDATA_INPUT_INDEX
    global CLASSES

    print('LOADING TF LITE MODEL...', end=' ')

    # Load TFLite model and allocate tensors.
    SCRIPT_DIRECTORY_PATH = os.path.abspath(os.path.dirname(__file__))
    interpreter = tflite.Interpreter(os.path.join(SCRIPT_DIRECTORY_PATH,"model/BirdNET_6K_GLOBAL_MODEL.tflite" ))
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input tensor index
    INPUT_LAYER_INDEX = input_details[0]['index']
    MDATA_INPUT_INDEX = input_details[1]['index']
    OUTPUT_LAYER_INDEX = output_details[0]['index']

    # Load labels
    CLASSES = []
    with open(os.path.join(SCRIPT_DIRECTORY_PATH,"model/labels.txt" ), 'r') as lfile:
        for line in lfile.readlines():
            CLASSES.append(line.replace('\n', ''))

    print('DONE!')

    return interpreter

def loadCustomSpeciesList(path):

    slist = []
    if os.path.isfile(path):
        with open(path, 'r') as csfile:
            for line in csfile.readlines():
                slist.append(line.replace('\r', '').replace('\n', ''))

    return slist

def splitSignal(sig, rate, overlap, seconds=3.0, minlen=1.5):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break
        
        # Signal chunk too short? Fill with zeros.
        if len(split) < int(rate * seconds):
            temp = np.zeros((int(rate * seconds)))
            temp[:len(split)] = split
            split = temp
        
        sig_splits.append(split)

    return sig_splits

def readAudioData(path, overlap, sample_rate=48000):

    print('READING AUDIO DATA...', end=' ', flush=True)
    print(f'Path: {path}')

    # Open file with librosa (uses ffmpeg or libav)
    try:
        sig, rate = librosa.load(path, sr=sample_rate, mono=True, res_type='kaiser_fast')
        clip_length = librosa.get_duration(y=sig, sr=rate)
    except Exception as exc:
        logging.exception(f'Error in loading {path} into librosa')
        return (0, 0)
    # Split audio into 3-second chunks
    chunks = splitSignal(sig, rate, overlap)

    print('DONE! READ', str(len(chunks)), 'CHUNKS.')

    return chunks, clip_length

def convertMetadata(m):

    # Convert week to cosine
    if m[2] >= 1 and m[2] <= 48:
        m[2] = math.cos(math.radians(m[2] * 7.5)) + 1 
    else:
        m[2] = -1

    # Add binary mask
    mask = np.ones((3,))
    if m[0] == -1 or m[1] == -1:
        mask = np.zeros((3,))
    if m[2] == -1:
        mask[2] = 0.0

    return np.concatenate([m, mask])

def custom_sigmoid(x, sensitivity=1.0):
    return 1 / (1.0 + np.exp(-sensitivity * x))

def predict(sample, interpreter, sensitivity, num_predictions):

    # Make a prediction
    interpreter.set_tensor(INPUT_LAYER_INDEX, np.array(sample[0], dtype='float32'))
    interpreter.set_tensor(MDATA_INPUT_INDEX, np.array(sample[1], dtype='float32'))
    interpreter.invoke()
    prediction = interpreter.get_tensor(OUTPUT_LAYER_INDEX)[0]

    # Apply custom sigmoid
    p_sigmoid = custom_sigmoid(prediction, sensitivity)

    # Get label and scores for pooled predictions
    p_labels = dict(zip(CLASSES, p_sigmoid))

    # Sort by score
    p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

    # Remove species that are on blacklist
    for i in range(min(num_predictions, len(p_sorted))):
        if p_sorted[i][0] in ['Human_Human', 'Non-bird_Non-bird', 'Noise_Noise']:
            p_sorted[i] = (p_sorted[i][0], 0.0)

    # Only return first the top ten results
    return p_sorted[:num_predictions]

def analyzeAudioData(chunks, lat, lon, week, sensitivity, overlap, interpreter, num_predictions):

    detections = {}
    start = time.time()
    print('ANALYZING AUDIO...', end=' ', flush=True)

    # Convert and prepare metadata
    mdata = convertMetadata(np.array([lat, lon, week]))
    mdata = np.expand_dims(mdata, 0)

    # Parse every chunk
    pred_start = 0.0
    for c in chunks:

        # Prepare as input signal
        sig = np.expand_dims(c, 0)

        # Make prediction
        p = predict([sig, mdata], interpreter, sensitivity, num_predictions)

        # Save result and timestamp
        pred_end = pred_start + 3.0
        detections[str(pred_start) + ';' + str(pred_end)] = p
        pred_start = pred_end - overlap

    print('DONE! Time', int((time.time() - start) * 10) / 10.0, 'SECONDS')

    return detections

def writeResultsToDf(df, detections, min_conf, output_metadata):

    rcnt = 0
    row = pd.DataFrame(output_metadata, index = [0])
    
    for d in detections:
        for entry in detections[d]:
            if entry[1] >= min_conf and (entry[0] in WHITE_LIST or len(WHITE_LIST) == 0):
                time_interval = d.split(';')
                row['OFFSET'] = float(time_interval[0])
                row['DURATION'] = float(time_interval[1])-float(time_interval[0])
                row['MANUAL ID'] = entry[0].split('_')[0]
                df = pd.concat([df,row], ignore_index=True)
                rcnt += 1
    print('DONE! WROTE', rcnt, 'RESULTS.')
    return df


def parseTestSet(path, file_type='wav'):

    # Find all soundscape files
    dataset = []
    if os.path.isfile(path):
        dataset.append(path)
    else:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                if f.rsplit('.', 1)[-1].lower() == file_type:
                    dataset.append(os.path.abspath(os.path.join(dirpath, f)))
    return dataset

def analyze(audio_path, output_path = None, lat=-1, lon=-1, week=-1, overlap=0.0,
    sensitivity=1.0, min_conf=0.1, custom_list='', filetype='wav', num_predictions=10,
    write_to_csv=False):
    
    global WHITE_LIST
    
    # Load model
    interpreter = loadModel()
    dataset = parseTestSet(audio_path, filetype)
    
    if not custom_list == '':
        WHITE_LIST = loadCustomSpeciesList(custom_list)
    else:
        WHITE_LIST = []
    
    # Write detections to output file
    min_conf = max(0.01, min(min_conf, 0.99))
    
    # Process audio data and get detections
    week = max(1, min(week, 48))
    sensitivity = max(0.5, min(1.0 - (sensitivity - 1.0), 1.5))
    sample_rate = 48000
    df_columns = {'FOLDER' : 'str', 'IN FILE' :'str', 'CLIP LENGTH' : 'float64', 'CHANNEL' : 'int64', 'OFFSET' : 'float64',
                'DURATION' : 'float64', 'SAMPLE RATE' : 'int64','MANUAL ID' : 'str'}
    df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in df_columns.items()})
    output_metadata = {}
    output_metadata['CHANNEL'] = 0 # Setting channel to 0 by default
    output_metadata['SAMPLE RATE'] = sample_rate
    output_file = os.path.join(audio_path, 'result.csv')

    if len(dataset) == 1:
        try:
            datafile = dataset[0]
            output_metadata['FOLDER']  = os.path.join('.', os.path.relpath(os.path.split(datafile)[0], os.getcwd())) + os.path.sep
            output_metadata['IN FILE'] =  os.path.split(datafile)[1]
            audioData, clip_length = readAudioData(datafile, overlap, sample_rate)
            output_metadata['CLIP LENGTH'] = clip_length
            detections = analyzeAudioData(audioData, lat, lon, week, sensitivity, overlap, interpreter, num_predictions)
            if output_path is None:
                output_file = os.path.join(output_metadata['FOLDER'], 'result.csv')
                output_file = os.path.abspath(output_file)
            else:
                output_directory = os.path.abspath(output_path) 
                if not os.path.exists(output_directory): 
                    os.makedirs(output_directory)
                output_file = os.path.join(output_directory, 'result.csv')
            df = writeResultsToDf(df, detections, min_conf, output_metadata)
        except KeyboardInterrupt:
            exit("Keyboard interrupt")
        except:
             print("Error processing file: {}".format(datafile))
    elif len(dataset) > 0:
        for datafile in dataset:         
            try:
                # Read audio data
                audioData, clip_length = readAudioData(datafile, overlap, sample_rate)
                if audioData == 0:
                    continue
                detections = analyzeAudioData(audioData, lat, lon, week, sensitivity, overlap, interpreter,  num_predictions)
                output_metadata['FOLDER']  = Path(datafile).resolve().as_posix()
                output_metadata['IN FILE'] = os.path.split(datafile)[1]
                output_metadata['CLIP LENGTH'] = clip_length
                df = writeResultsToDf(df, detections, min_conf, output_metadata)
            except KeyboardInterrupt:
                exit("Keyboard interrupt")
            except Exception as exc:
                print("Error in processing file: {}".format(datafile)) 
                logging.exception(f'Error in processing {datafile}')
        if output_path is None:
            output_file = os.path.join(audio_path, 'result.csv')
            output_file = os.path.abspath(output_file)
        else:
            output_directory = os.path.abspath(output_path) 
            if not os.path.exists(output_directory): 
                os.makedirs(output_directory)
            output_file = os.path.join(output_directory, 'result.csv')
    else:
        print("No input file/folder passed")
        exit()
    if write_to_csv:
        print('WRITING RESULTS TO', output_file, '...', end=' ')
        df.to_csv(output_file, index=False)
    return df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', help='Path to input folder/input file. All the nested folders will also be processed.')
    parser.add_argument('--o', default='result.csv', help='Absolute path to output folder. By default results are written into the input folder.')
    parser.add_argument('--lat', type=float, default=-1, help='Recording location latitude. Set -1 to ignore.')
    parser.add_argument('--lon', type=float, default=-1, help='Recording location longitude. Set -1 to ignore.')
    parser.add_argument('--week', type=int, default=-1, help='Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 to ignore.')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap in seconds between extracted spectrograms. Values in [0.0, 2.9]. Defaults tp 0.0.')
    parser.add_argument('--sensitivity', type=float, default=1.0, help='Detection sensitivity; Higher values result in higher sensitivity. Values in [0.5, 1.5]. Defaults to 1.0.')
    parser.add_argument('--min_conf', type=float, default=0.1, help='Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.')   
    parser.add_argument('--custom_list', default='', help='Path to text file containing a list of species. Not used if not provided.')
    parser.add_argument('--filetype', default='wav', help='Filetype of soundscape recordings. Defaults to \'wav\'.')
    parser.add_argument('--num_predictions', type=int, default=10, help='Defines maximum number of written predictions in a given 3s segment. Defaults to 10')
    args = parser.parse_args()

    df = analyze(
        audio_path=args.i, output_path=args.o, lat=args.lat, lon=args.lon,
        week=args.week, overlap=args.overlap, sensitivity=args.sensitivity,
        min_conf=args.min_conf, custom_list=args.custom_list, filetype=args.filetype,
        num_predictions=args.num_predictions, write_to_csv=True
    )  

if __name__ == '__main__':

    main()

    # Example calls
    # python3 analyze.py --i 'example/XC558716 - Soundscape.mp3' --lat 35.4244 --lon -120.7463 --week 18
    # python3 analyze.py --i 'example/XC563936 - Soundscape.mp3' --lat 47.6766 --lon -122.294 --week 11 --overlap 1.5 --min_conf 0.25 --sensitivity 1.25 --custom_list 'example/custom_species_list.txt'
