from PyHa.statistics import *
from PyHa.IsoAutio import *
from PyHa.visualizations import *
import pandas as pd
import timeit

def temporal_performance(audio_path,labels_path):

    # Dictionary that will be used to construct the output Pandas Dataframe
    output_df = {
        "FUNCTION": [],
        "TECHNIQUE": [],
        "DURATION": [],
        "ITERATIONS": []
    }
    setup1 = '''

from PyHa.IsoAutio import generate_automated_labels

import pandas as pd
import timeit
isolation_parameters = {
    "technique" : "steinberg",
    "threshold_type" : "median",
    "threshold_const" : 2.0,
    "threshold_min" : 0.0,
    "bi_directional_jump" : 1.0,
    "chunk_size" : 5.0
}
'''+ "path = \"" + audio_path + "\""

    code1 = '''automated_df = generate_automated_labels(path,isolation_parameters)'''
    test1 = timeit.timeit(setup=setup1,stmt=code1,number=10)
    output_df["FUNCTION"].append("generate_automated_labels()")
    output_df["TECHNIQUE"].append("steinberg")
    output_df["DURATION"].append(test1)
    output_df["ITERATIONS"].append(10)
    setup2 = '''

from PyHa.IsoAutio import generate_automated_labels

import pandas as pd
import timeit
isolation_parameters = {
    "technique" : "simple",
    "threshold_type" : "median",
    "threshold_const" : 2.0,
    "threshold_min" : 0.0,
    "bi_directional_jump" : 1.0,
    "chunk_size" : 5.0
}


'''+ "path = \"" + audio_path + "\""

    code2 = '''automated_df = generate_automated_labels(path,isolation_parameters)'''
    test2 = timeit.timeit(setup=setup2,stmt=code2,number=10)
    output_df["FUNCTION"].append("generate_automated_labels()")
    output_df["TECHNIQUE"].append("simple")
    output_df["DURATION"].append(test2)
    output_df["ITERATIONS"].append(10)

    setup3 = '''

from PyHa.IsoAutio import generate_automated_labels

import pandas as pd
import timeit
isolation_parameters = {
    "technique" : "stack",
    "threshold_type" : "median",
    "threshold_const" : 2.0,
    "threshold_min" : 0.0,
    "bi_directional_jump" : 1.0,
    "chunk_size" : 5.0
}

''' + "path = \"" + audio_path + "\""

    code3 = '''automated_df = generate_automated_labels(path,isolation_parameters)'''
    test3 = timeit.timeit(setup=setup3,stmt=code3,number=10)
    output_df["FUNCTION"].append("generate_automated_labels()")
    output_df["TECHNIQUE"].append("stack")
    output_df["DURATION"].append(test3)
    output_df["ITERATIONS"].append(10)
    setup4 = '''

from PyHa.IsoAutio import generate_automated_labels

import pandas as pd
import timeit
isolation_parameters = {
    "technique" : "chunk",
    "threshold_type" : "median",
    "threshold_const" : 2.0,
    "threshold_min" : 0.0,
    "bi_directional_jump" : 1.0,
    "chunk_size" : 5.0
}

''' + "path = \"" + audio_path + "\""
    code4 = '''automated_df = generate_automated_labels(path,isolation_parameters)'''
    test4 = timeit.timeit(setup=setup4,stmt=code4,number=10)

    output_df["FUNCTION"].append("generate_automated_labels()")
    output_df["TECHNIQUE"].append("chunk")
    output_df["DURATION"].append(test4)
    output_df["ITERATIONS"].append(10)

    setup5 = '''

from PyHa.IsoAutio import generate_automated_labels
from PyHa.statistics import bird_label_scores

import pandas as pd
import timeit
isolation_parameters = {
    "technique" : "chunk",
    "threshold_type" : "median",
    "threshold_const" : 2.0,
    "threshold_min" : 0.0,
    "bi_directional_jump" : 1.0,
    "chunk_size" : 5.0
}

''' + "path = \"" + audio_path + "\"" + "\nmanual_df = pd.read_csv(" +"\"" + labels_path + "\")"

    generate_labels = '''automated_df = generate_automated_labels(path,isolation_parameters)'''
    setup5 = setup5 + '\n' + generate_labels
    isolate_clip = '''\nclip_name = automated_df["IN FILE"][0]
test_automated_df = automated_df[automated_df["IN FILE"] == clip_name]
test_manual_df = manual_df[manual_df["IN FILE"] == clip_name]'''
    setup5 = setup5 + isolate_clip
    code5 = '''stats = bird_label_scores(test_automated_df,test_manual_df)'''
    test5 = timeit.timeit(setup=setup5,stmt=code5,number=10)

    output_df["FUNCTION"].append("bird_label_scores()")
    output_df["TECHNIQUE"].append("chunk")
    output_df["DURATION"].append(test5)
    output_df["ITERATIONS"].append(10)

    setup6 = '''

from PyHa.IsoAutio import generate_automated_labels
from PyHa.statistics import automated_labeling_statistics

import pandas as pd
import timeit
isolation_parameters = {
    "technique" : "chunk",
    "threshold_type" : "median",
    "threshold_const" : 2.0,
    "threshold_min" : 0.0,
    "bi_directional_jump" : 1.0,
    "chunk_size" : 5.0
}

''' + "path = \"" + audio_path + "\"" + "\nmanual_df = pd.read_csv(" +"\"" + labels_path + "\")"

    generate_labels = '''automated_df = generate_automated_labels(path,isolation_parameters)'''
    setup6 = setup6 + '\n' + generate_labels

    code6 = '''statistics_df = automated_labeling_statistics(automated_df,manual_df,stats_type = "general")'''
    test6 = timeit.timeit(setup=setup6,stmt=code6,number=10)

    output_df["FUNCTION"].append("automated_labeling_statistics()")
    output_df["TECHNIQUE"].append("General Overlap")
    output_df["DURATION"].append(test6)
    output_df["ITERATIONS"].append(10)

    setup7 = '''

from PyHa.IsoAutio import generate_automated_labels
from PyHa.statistics import clip_IoU

import pandas as pd
import timeit
isolation_parameters = {
    "technique" : "chunk",
    "threshold_type" : "median",
    "threshold_const" : 2.0,
    "threshold_min" : 0.0,
    "bi_directional_jump" : 1.0,
    "chunk_size" : 5.0
}

''' + "path = \"" + audio_path + "\"" + "\nmanual_df = pd.read_csv(" +"\"" + labels_path + "\")"

    generate_labels = '''automated_df = generate_automated_labels(path,isolation_parameters)'''
    setup7 = setup7 + '\n' + generate_labels
    isolate_clip = '''\nclip_name = automated_df["IN FILE"][0]
test_automated_df = automated_df[automated_df["IN FILE"] == clip_name]
test_manual_df = manual_df[manual_df["IN FILE"] == clip_name]'''
    setup7 = setup7 + isolate_clip
    code7 = '''IoU_Matrix = clip_IoU(test_automated_df,test_manual_df)'''
    test7 = timeit.timeit(setup=setup7,stmt=code7,number=10)

    output_df["FUNCTION"].append("clip_IoU")
    output_df["TECHNIQUE"].append("IoU")
    output_df["DURATION"].append(test7)
    output_df["ITERATIONS"].append(10)

    setup8 = '''

from PyHa.IsoAutio import generate_automated_labels
from PyHa.statistics import clip_IoU
from PyHa.statistics import matrix_IoU_Scores

import pandas as pd
import timeit
isolation_parameters = {
    "technique" : "chunk",
    "threshold_type" : "median",
    "threshold_const" : 2.0,
    "threshold_min" : 0.0,
    "bi_directional_jump" : 1.0,
    "chunk_size" : 5.0
}

''' + "path = \"" + audio_path + "\"" + "\nmanual_df = pd.read_csv(" +"\"" + labels_path + "\")"

    generate_labels = '''automated_df = generate_automated_labels(path,isolation_parameters)'''
    setup8 = setup8 + '\n' + generate_labels
    isolate_clip = '''\nclip_name = automated_df["IN FILE"][0]
test_automated_df = automated_df[automated_df["IN FILE"] == clip_name]
test_manual_df = manual_df[manual_df["IN FILE"] == clip_name]
IoU_Matrix = clip_IoU(test_automated_df,test_manual_df)'''
    setup8 = setup8 + isolate_clip
    code8 = '''scores = matrix_IoU_Scores(IoU_Matrix,test_manual_df,0.5)'''
    test8 = timeit.timeit(setup=setup8,stmt=code8,number=10)
    output_df["FUNCTION"].append("matrix_IoU_Scores")
    output_df["TECHNIQUE"].append("IoU")
    output_df["DURATION"].append(test8)
    output_df["ITERATIONS"].append(10)

    setup9 = '''

from PyHa.IsoAutio import generate_automated_labels
from PyHa.statistics import automated_labeling_statistics

import pandas as pd
import timeit
isolation_parameters = {
    "technique" : "chunk",
    "threshold_type" : "median",
    "threshold_const" : 2.0,
    "threshold_min" : 0.0,
    "bi_directional_jump" : 1.0,
    "chunk_size" : 5.0
}

''' + "path = \"" + audio_path + "\"" + "\nmanual_df = pd.read_csv(" +"\"" + labels_path + "\")"

    generate_labels = '''automated_df = generate_automated_labels(path,isolation_parameters)'''
    setup9 = setup9 + '\n' + generate_labels

    code9 = '''statistics_df = automated_labeling_statistics(automated_df,manual_df,stats_type = "IoU")'''
    test9 = timeit.timeit(setup=setup9,stmt=code9,number=2)

    output_df["FUNCTION"].append("automated_labeling_statistics")
    output_df["TECHNIQUE"].append("IoU")
    output_df["DURATION"].append(test9)
    output_df["ITERATIONS"].append(2)

    output_df = pd.DataFrame(output_df)
    output_df["AVERAGE"] = output_df["DURATION"]/output_df["ITERATIONS"]
    return output_df
