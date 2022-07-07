import pandas as pd
import numpy as np


def annotation_chunker(kaleidoscope_df, chunk_length):
    """
    Function that converts a Kaleidoscope-formatted Dataframe containing 
    annotations to uniform chunks of chunk_length.

    Note: if all or part of an annotation covers the last < chunk_length
    seconds of a clip it will be ignored. If two annotations overlap in 
    the same 3 second chunk, both are represented in that chunk

    Args:
        kaleidoscope_df (Dataframe)
            - Dataframe of annotations in kaleidoscope format

        chunk_length (int)
            - duration to set all annotation chunks
    Returns:
        Dataframe of labels with chunk_length duration 
        (elements in "OFFSET" are divisible by chunk_length).
    """

    #Init list of clips to cycle through and output dataframe
    kaleidoscope_df["FILEPATH"] =  kaleidoscope_df["FOLDER"] + kaleidoscope_df["IN FILE"] 
    clips = kaleidoscope_df["FILEPATH"].unique()
    df_columns = {'FOLDER': 'str', 'IN FILE' :'str', 'CLIP LENGTH' : 'float64', 'CHANNEL' : 'int64', 'OFFSET' : 'float64',
                'DURATION' : 'float64', 'SAMPLE RATE' : 'int64','MANUAL ID' : 'str'}
    output_df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in df_columns.items()})
    
    # going through each clip
    for clip in clips:
        clip_df = kaleidoscope_df[kaleidoscope_df["FILEPATH"] == clip]
        path = clip_df["FOLDER"].unique()[0]
        file = clip_df["IN FILE"].unique()[0]
        birds = clip_df["MANUAL ID"].unique()
        sr = clip_df["SAMPLE RATE"].unique()[0]
        clip_len = clip_df["CLIP LENGTH"].unique()[0]

        # quick data sanitization to remove very short clips
        # do not consider any chunk that is less than chunk_length
        if clip_len < chunk_length:
            continue
        potential_annotation_count = int(clip_len)//int(chunk_length)

        # going through each species that was ID'ed in the clip
        arr_len = int(clip_len*1000)
        for bird in birds:
            species_df = clip_df[clip_df["MANUAL ID"] == bird]
            human_arr = np.zeros((arr_len))
            # looping through each annotation
            for annotation in species_df.index:
                minval = int(round(species_df["OFFSET"][annotation] * 1000, 0))
                # Determining the end of a human label
                maxval = int(
                    round(
                        (species_df["OFFSET"][annotation] +
                         species_df["DURATION"][annotation]) *
                        1000,
                        0))
                # Placing the label relative to the clip
                human_arr[minval:maxval] = 1
            # performing the chunk isolation technique on the human array

            for index in range(potential_annotation_count):
                chunk_start = index * (chunk_length*1000)
                chunk_end = min((index+1)*chunk_length*1000,arr_len)
                chunk = human_arr[int(chunk_start):int(chunk_end)]
                if max(chunk) >= 0.5:
                    row = pd.DataFrame(index = [0])
                    annotation_start = chunk_start / 1000
                    #updating the dictionary
                    row["FOLDER"] = path
                    row["IN FILE"] = file
                    row["CLIP LENGTH"] = clip_len
                    row["OFFSET"] = annotation_start
                    row["DURATION"] = chunk_length
                    row["SAMPLE RATE"] = sr
                    row["MANUAL ID"] = bird
                    row["CHANNEL"] = 0
                    output_df = pd.concat([output_df,row], ignore_index=True)
    return output_df