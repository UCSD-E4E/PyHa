"""
Evaluator class
Evaluates one single runfile
_evaluate method is called by the AICrowd framework and returns an object holding up to 2 different scores
"""

import csv
import datetime
from operator import itemgetter

chunk_duration = 5 #seconds

class BirdSoundscapeEvaluator:

    """
    Constructor
    Parameter 'answer_file_path': Path of file containing ground truth
    """
    def __init__(self, answer_file_path, allowed_classes_file_path):
        #Ground truth file
        self.answer_file_path = answer_file_path

        #Ground truth data
        self.gt = self.load_gt()

        #allowed ids in the predictions files
        self.allowed_classes_file_path = allowed_classes_file_path

    """
    This is the only method that will be called by the framework
    Parameter 'submission_file_path': Path of the submitted runfile
    returns a _result_object that can contain up to 2 different scores
    """
    def _evaluate(self, submission_file_path):
        #Load predictions
        predictions = self.load_predictions(submission_file_path)

        if predictions != None:
            #Compute first score
            cmap = self.classification_mean_average_precision(predictions)
            #Compute second score
            rmap = self.retrieval_mean_average_precision(predictions)

            #Create object that is returned to the CrowdAI framework
            _result_object = {
                "classification_map": cmap,
                "retrieval_map" : rmap
            }

            return _result_object


    """
    Load and return groundtruth data
    """
    def load_gt(self):
        gt = {}
        gt['by_class'] = {}
        gt['by_query'] = {}

        with open(self.answer_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
            for row in reader:
                query = row[0]
                classid = row[1]
                timecodes = row[2]
                #country = row[3]

                timecoded_query_split_into_chunks = []
                chunks = self.timecodes_to_chunks(timecodes, chunk_duration)

                for chunk in chunks:
                    query_tc = query + '_' + chunk

                    if not classid in gt['by_class']:
                        gt['by_class'][classid] = set()
                    gt['by_class'][classid].add(query_tc)

                    if not query_tc in gt['by_query']:
                        gt['by_query'][query_tc] = set()
                    gt['by_query'][query_tc].add(classid)

        return gt


    def timecodes_to_chunks(self, tcs, second_base):
        resolution=datetime.timedelta(seconds=second_base)
        chunk_duration=datetime.timedelta(hours=0,minutes=0,seconds=second_base)

        tcstart = tcs.split('-')[0]
        tcend = tcs.split('-')[1]

        hs = int(tcstart.split(':')[0])
        ms = int(tcstart.split(':')[1])
        ss = int(tcstart.split(':')[2])
        hmss=datetime.timedelta(hours=hs,minutes=ms,seconds=ss)
        modulos = datetime.timedelta(seconds=hmss.seconds%resolution.seconds)
        tcstartrounded= hmss - modulos

        he = int(tcend.split(':')[0])
        me = int(tcend.split(':')[1])
        se = int(tcend.split(':')[2])
        hmse=datetime.timedelta(hours=he,minutes=me,seconds=(se))
        hmse_extended = hmse + chunk_duration
        moduloe = datetime.timedelta(seconds=hmse_extended.seconds%resolution.seconds)
        tcendrounded = hmse_extended - moduloe

        chunks = []
        current_s = tcstartrounded
        current_e = current_s + chunk_duration
        while(current_e  <= tcendrounded and current_s < hmse):
            start_seconds = current_s.total_seconds()
            formated_start = datetime.datetime.utcfromtimestamp(start_seconds).strftime("%H:%M:%S")

            end_seconds = current_e.total_seconds()
            formated_end = datetime.datetime.utcfromtimestamp(end_seconds).strftime("%H:%M:%S")
            chunk = formated_start+'-'+formated_end
            chunks.append(chunk)
            current_e += chunk_duration
            current_s += chunk_duration

        return chunks


    """
    Load and return allowed class ids in the predictions files
    """
    def load_allowed_classes(self):
        #...
        #return gt
        allowed_classes = set()
        with open(self.allowed_classes_file_path) as f:
            for classid in f.readlines():
                allowed_classes.add(classid.rstrip("\n"))
        return allowed_classes


    """
    Loads and returns a predictions object (dictionary) that contains the submitted data that will be used in the _evaluate method
    Parameter 'submission_file_path': Path of the submitted runfile
    Validation of the runfile format will also be handled here
    """

    def load_predictions(self, submission_file_path):
        #...
        #returns predictions
        class_to_querytc_score_list = {}
        querytc_to_classid_score_list = {}
        predictions = {}
        predictions['by_class'] = class_to_querytc_score_list
        predictions['by_query'] = querytc_to_classid_score_list


        allowed_query_tcs = self.gt['by_query'].keys()

        allowed_query_ids = set([query_tc.split('_')[0] for query_tc in allowed_query_tcs])

        allowed_classes = self.load_allowed_classes()

        max_propositions = 10 #max nbr of classes for query_tc

        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
            lineCnt = 0

            for row in reader:
                lineCnt += 1

                if len(row) != 4:
                    raise Exception("Wrong format: Each line must consist of a Media ID, TimeCodeStart-TimeCodeEnd, class ID, probability separated by semicolons ({}) {}"
                        .format("<MediaId>;<TimeCodeStart-TimeCodeEnd><ClassId><Probability>", self.line_nbr_string(lineCnt)))

                query_id = row[0]

                timecodes = row[1]
                self.check_time_interval(timecodes, lineCnt)
                if query_id not in allowed_query_ids:
                    raise Exception("MediaID '{}' in submission file does not exist in testset {}"
                        .format(query_id, self.line_nbr_string(lineCnt)))

                query_tc = query_id + '_' + timecodes

                class_id = row[2]
                # Class ID not in testset => Error
                if class_id not in allowed_classes:
                    raise Exception("'{}' is not a valid class ID {}"
                        .format(class_id, self.line_nbr_string(lineCnt)))


                # 4th value in line is not a number or not between 0 and 1 => Error
                try:
                    probability = float(row[3])
                    if probability < 0 or probability > 1:
                        raise ValueError
                except ValueError:
                    raise Exception("Score must be a probability between 0 and 1 {}"
                        .format(self.line_nbr_string(lineCnt)))


                querytc_score_list = []
                if not class_id in class_to_querytc_score_list:
                    class_to_querytc_score_list[class_id] = querytc_score_list
                else:
                    querytc_score_list = class_to_querytc_score_list[class_id]

                #for managing equiproba cases later
                correct_prediction = 0
                if class_id in self.gt['by_class']:
                    if query_tc in self.gt['by_class'][class_id]:
                        correct_prediction = 1

                querytc_score_list.append([query_tc, probability, correct_prediction])

                classid_score_list = querytc_to_classid_score_list.get(query_tc, list())
                occured_class_ids = [item[0] for item in classid_score_list]
                if class_id in occured_class_ids:
                    raise Exception("Prediction for chunk {} already exists, {}"
                        .format(query_tc, self.line_nbr_string(lineCnt)))

                #for managing equiproba cases later
                correct_prediction = 0
                if query_tc in self.gt['by_query']:
                    if class_id in self.gt['by_query'][query_tc]:
                        correct_prediction = 1

                classid_score_list.append([class_id, probability, correct_prediction])

                querytc_to_classid_score_list[query_tc] = classid_score_list

                if len(querytc_to_classid_score_list[query_tc]) > max_propositions:
                    raise Exception("There are more than 100 propositions for chunck {}, {}"
                        .format(query_tc, self.line_nbr_string(lineCnt)))

        return predictions


    def check_time_interval(self, time_interval, lineCnt):
        # Timespan not consisting of 2 tokens separated by '-' => Error
        times = time_interval.split("-")
        if len(times) != 2:
            raise Exception("A time interval consisting of 2 timecodes separated by a '-' must be provided ('hh:mm:ss-hh:mm:ss'). {}"
                .format(self.line_nbr_string(lineCnt)))

        #Cannot parse datetime => Error
        try:
            datetime_1 = datetime.datetime.strptime(times[0], "%H:%M:%S")
            datetime_2 = datetime.datetime.strptime(times[1], "%H:%M:%S")
        except :
            raise Exception("Time code in time interval cannot be parsed. It must have the following format: 'hh:mm:ss'. {}"
                .format(self.line_nbr_string(lineCnt)))

        hms_1 = times[0].split(":")
        hms_2 = times[1].split(":")
        seconds_1 = int(hms_1[2])
        seconds_2 = int(hms_2[2])

        #Seconds not multiples of 5? => Error
        if ((seconds_1 % chunk_duration) != 0) or ((seconds_2 % chunk_duration) != 0):
            raise Exception("Time code must be a 'multiple' of {} seconds.  {}"
                .format(chunk_duration, self.line_nbr_string(lineCnt)))

        delta_5_s = datetime.timedelta(seconds=chunk_duration)
        expected_datetime_2 = datetime_1 + delta_5_s

        if datetime_2 != expected_datetime_2:
            raise Exception("Time interval must have a duration of 5 seconds. {}"
                .format(self.line_nbr_string(lineCnt)))

    def line_nbr_string(self, line_nbr):
        return "(Line nbr {})".format(line_nbr)



    """
    Compute and return the primary score
    Parameter 'predictions' : predictions object generated by the load_predictions method
    We assume that the predictions in the parameter are valid
    Valiation should be handled in the load_predictions method
    """
    def classification_mean_average_precision(self, predictions):
        return self.compute_map_score('by_class', predictions)


    """
    Compute and return the secondary score
    Parameter 'predictions' : predictions object generated by the load_predictions method
    We assume that the predictions in the parameter are valid
    Valiation should be handled in the load_predictions method
    """
    def retrieval_mean_average_precision(self, predictions):
        return self.compute_map_score('by_query', predictions)



    def compute_map_score(self, by_type, predictions):
        map = 0.0
        for sample in self.gt[by_type]:
            ap = 0.0
            count_relevant = 0
            rank = 0
            if sample in predictions[by_type]:
                score_list = predictions[by_type][sample]
                score_list_sorted_reversed = list(reversed(sorted(score_list, key=itemgetter(1,2))))

                for proposition in score_list_sorted_reversed:
                    rank += 1
                    label= proposition[0]
                    if label in self.gt[by_type][sample]:
                        count_relevant += 1
                        ap +=  float(count_relevant) / float(rank)
                ap = ap / float(len(self.gt[by_type][sample]))

            map += ap

        map =  map / float (len(self.gt[by_type]))

        return map


        

"""
Test evaluation a runfile
provide path to ground truth file in constructor
call _evaluate method with path of submitted file as argument
"""
if __name__ == "__main__":

  #Ground truth file
  gt_file_path = "gt_birdclef2020_validation_data.csv"

  #allowed classids
  allowed_classes_file_path = 'allowed_classes.txt'

  #Submission file
  submission_file_path = "perfect_run.csv"

  #Create instance of Evaluator
  evaluator = BirdSoundscapeEvaluator(gt_file_path, allowed_classes_file_path)
  
  #Call _evaluate method
  result = evaluator._evaluate(submission_file_path)
  print(result)

  
