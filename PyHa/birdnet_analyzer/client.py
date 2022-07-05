import os
import json
import time
import requests
import argparse

from multiprocessing import freeze_support

def sendRequest(host, port, fpath, mdata):

    url = 'http://{}:{}/analyze'.format(host, port)

    print('Requesting analysis for {}'.format(fpath))

    # Make payload
    multipart_form_data = {
        'audio': (fpath.split(os.sep)[-1], open(fpath, 'rb')),
        'meta': (None, mdata)
    }

    # Send request
    start_time = time.time()
    response = requests.post(url, files=multipart_form_data)
    end_time = time.time()

    print('Response: {}, Time: {:.4f}s'.format(response.text, end_time - start_time), flush=True)

    # Convert to dict
    data = json.loads(response.text)
    
    return data

def saveResult(data, fpath):

    # Make directory 
    dir_path = os.path.dirname(fpath)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Save result
    with open(fpath, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':

    # Freeze support for excecutable
    freeze_support()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Client that queries an analyzer API endpoint server.')
    parser.add_argument('--host', default='localhost', help='Host name or IP address of API endpoint server.')   
    parser.add_argument('--port', type=int, default=8080, help='Port of API endpoint server.')     
    parser.add_argument('--i', default='example/soundscape.wav', help='Path to file that should be analyzed.')  
    parser.add_argument('--o', default='', help='Path to result file. Leave blank to store with audio file.')   
    parser.add_argument('--lat', type=float, default=-1, help='Recording location latitude. Set -1 to ignore.')
    parser.add_argument('--lon', type=float, default=-1, help='Recording location longitude. Set -1 to ignore.')
    parser.add_argument('--week', type=int, default=-1, help='Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 for year-round species list.')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap of prediction segments. Values in [0.0, 2.9]. Defaults to 0.0.')
    parser.add_argument('--sensitivity', type=float, default=1.0, help='Detection sensitivity; Higher values result in higher sensitivity. Values in [0.5, 1.5]. Defaults to 1.0.')
    parser.add_argument('--pmode', default='avg', help='Score pooling mode. Values in [\'avg\', \'max\']. Defaults to \'avg\'.')   
    parser.add_argument('--num_results', type=int, default=5, help='Number of results per request. Defaults to 5.')
    parser.add_argument('--sf_thresh', type=float, default=0.03, help='Minimum species occurrence frequency threshold for location filter. Values in [0.01, 0.99]. Defaults to 0.03.')
    parser.add_argument('--save', type=bool, default=False, help='Define if files should be stored on server. Values in [True, False]. Defaults to False.')    

    args = parser.parse_args()

    # Make metadata
    mdata = {'lat': args.lat, 
             'lon': args.lon, 
             'week': args.week,
             'overlap': args.overlap,
             'sensitivity': args.sensitivity,
             'sf_thresh': args.sf_thresh,
             'pmode': args.pmode,
             'num_results': args.num_results,
             'save': args.save}

    # Send request
    data = sendRequest(args.host, args.port, args.i, json.dumps(mdata))

    # Save result
    if len(args.o) > 0:
        fpath = args.o
    else:
        fpath = args.i.rsplit('.', 1)[0] + '.BirdNET.results.json'
    saveResult(data, fpath)

    # A few examples to test
    # python3 client.py --host localhost --port 8080 --i example/soundscape.wav
    # python3 client.py --host localhost --port 8080 --i example/soundscape.wav --save True --lat 42.5 --lon -76.45 --week 4
    # python3 client.py --host localhost --port 8080 --i example/soundscape.wav --save True --lat 42.5 --lon -76.45 --week 4 --overlap 2.5 --sensitivity 1.25
    # python3 client.py --host localhost --port 8080 --i example/soundscape.wav --save True --lat 42.5 --lon -76.45 --week 4 --pmode max