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
    clips = kaleidoscope_df["IN FILE"].unique()
    df_columns = {'IN FILE' :'str', 'CLIP LENGTH' : 'float64', 'CHANNEL' : 'int64', 'OFFSET' : 'float64',
                'DURATION' : 'float64', 'SAMPLE RATE' : 'int64','MANUAL ID' : 'str'}
    output_df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in df_columns.items()})
    
    # going through each clip
    for clip in clips:
        clip_df = kaleidoscope_df[kaleidoscope_df["IN FILE"] == clip]
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
                    row["IN FILE"] = clip
                    row["CLIP LENGTH"] = clip_len
                    row["OFFSET"] = annotation_start
                    row["DURATION"] = chunk_length
                    row["SAMPLE RATE"] = sr
                    row["MANUAL ID"] = bird
                    row["CHANNEL"] = 0
                    output_df = pd.concat([output_df,row], ignore_index=True)
    return output_df

def create_chunk_row(row, rows_to_add, new_start, duration):
    """
    Helper function that takes in a Dataframe containing annotations 
    and appends a single row to the Dataframe before returning it.

    Args:
        row (Series)
            - Row of a single annotation

        rows_to_add (Dataframe)
            - Dataframe of labels
        
        new_start (float)
            - The start time of the annotation in row

        duration (int)
            - The duration of the annotation in row
    Returns:
        Dataframe of labels with the newly appended row
    """
    chunk_row = row.copy()
    chunk_row["DURATION"] = duration
    chunk_row["OFFSET"] = new_start
    rows_to_add.append(chunk_row.to_frame().T)
    return rows_to_add

def convolving_chunk(row, chunk_length=3, min_length=0.4, only_slide=False):
    """
    Helper function that converts a Dataframe row containing a binary
    annotation to uniform chunks of chunk_length. 

    Note: Annotations of length shorter than min_length are ignored. Annotations
    that are shorter than or equal to chunk_length are chopped into three chunks
    where the annotation is placed at the start, middle, and end. Annotations
    that are longer than chunk_length are chunked used a sliding window.

    Args:
        row (Series)
            - Row of a single annotation

        chunk_length (int)
            - duration in seconds to set all annotation chunks
        
        min_length (float)
            - duration in seconds to ignore annotations shorter in length

        only_slide (bool)
            - If True, only annotations greater than chunk_length are chunked
    Returns:
        Dataframe of labels with chunk_length duration 
        (elements in "OFFSET" are divisible by chunk_length).
    """
    chunk_df = pd.DataFrame(columns=row.to_frame().T.columns)
    rows_to_add = []
    offset = row["OFFSET"]
    duration = row["DURATION"]
    chunk_half_duration = chunk_length / 2
    
    #Ignore small duration (could be errors, play with this value)
    if duration < min_length:
        return chunk_df
    
    if duration <= chunk_length and not only_slide:
        #Put the original bird call at...
        #1) Start of clip
        if offset+chunk_length < row["CLIP LENGTH"]:
            rows_to_add = create_chunk_row(row, rows_to_add, offset, chunk_length)
            
        #2) End of clip
        if offset+duration-chunk_length > 0: 
            rows_to_add = create_chunk_row(row, rows_to_add, offset+duration-chunk_length, chunk_length)
            
        #3) Middle of clip
        if offset+duration-chunk_half_duration>0 and (offset+duration+chunk_half_duration) < row["CLIP LENGTH"]:
            
            #Could be better placed in middle, maybe with some randomization?
            rows_to_add = create_chunk_row(row, rows_to_add, (offset+duration-chunk_half_duration), chunk_length)
            
    
    #Longer than chunk duration
    else:
        #Perform Yan's Sliding Window operation
        clip_num=int(duration/(chunk_half_duration))
        for i in range(clip_num-1):
            new_start = offset+i*chunk_half_duration
            new_end = offset + chunk_length+i*chunk_half_duration
            if new_end < row["CLIP LENGTH"]:
                rows_to_add = create_chunk_row(row, rows_to_add, new_start, chunk_length) 
    
    #Add all new rows to our return df
    if len(rows_to_add) == 0:
        return chunk_df
    
    chunk_df = pd.concat(rows_to_add,  ignore_index=True)

    return chunk_df

def dynamic_yan_chunking(df, chunk_length=3, min_length=0.4, only_slide=False):
    """
    Function that converts a Dataframe containing binary
    annotations to uniform chunks of chunk_length. 

    Note: Annotations shorter than min_length are ignored. Annotations
    shorter than or equal to chunk_length are chopped into three chunks
    where the annotation is placed at the start, middle, and end. Annotations
    longer than chunk_length are chunked used a sliding window.

    Args:
        df (Dataframe)
            - Dataframe of annotations 

        chunk_length (int)
            - duration in seconds to set all annotation chunks
        
        min_length (float)
            - duration in seconds to ignore annotations shorter than

        only_slide (bool)
            - If True, only annotations greater than chunk_length are chunked
    Returns:
        Dataframe of labels with chunk_length duration 
        (elements in "OFFSET" are divisible by chunk_length).
    """
    return_df = pd.DataFrame(columns=df.columns)
    
    for _, row in df.iterrows():
        chunk_df = convolving_chunk(row, min_length=min_length, chunk_length=chunk_length, only_slide=only_slide)
        return_df = pd.concat([return_df, chunk_df],  ignore_index=True)
    
    return return_df