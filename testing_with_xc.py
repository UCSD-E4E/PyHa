import pandas as pd

labels = pd.read_csv("for_data_science_newline_fixed.csv")
uniqueFileNames = labels["IN FILE"].unique()
sampleFiles = pd.Series(uniqueFileNames).sample(5)
print(sampleFiles)

# Files to use
## Monasa-morphoeus-257977.wav
## Agelasticus-xanthophthalmus-20921.wav
## Pipile-cumanensis-257420.wav
## Setophaga-palmarum-173794.wav
## Mycteria-americana-193137.wav --> Took off this one since it is 216 s, so clip_IoU took too long
