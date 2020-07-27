import pandas as pd
import sys

print(sys.argv)

df = pd.read_csv(sys.argv[1], index_col=False)


missing = [i for i in range(103)]

for idx in df.index:
    run = df.loc[idx, 'case name']
    run = run.replace('Run_','')
    run = int(run[:3:])
    if run in missing:
        missing.remove(run)

out = "for JOB in "
for m in missing:
    out += f"{m} "
out = out.strip()
out += "; do"
print(out)
