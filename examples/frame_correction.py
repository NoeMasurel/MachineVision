import pandas as pd
INPUT_FILE = "gt.txt"

start_range = int(input("\nStart frame : "))
decalage = 160 # int(input("\nDecalage :"))


df = pd.read_csv(INPUT_FILE, header=None)

# if start_range and end_range:
#     df = df[df[0].between(start_range, end_range)]

if decalage:
    df[0] -= decalage

df.to_csv(f"{INPUT_FILE}", index=False, header=False)

