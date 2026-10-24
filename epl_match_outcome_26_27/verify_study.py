import os,json
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
root=Path(__file__).resolve().parent
os.chdir(root)
import IPython.display
IPython.display.display=lambda *a,**k: None
import matplotlib.pyplot as plt
plt.show=lambda *a,**k:None
nb=json.load(open(root/'notebooks/epl_match_outcome_prediction.ipynb'))
ns={'__name__':'__main__'}
for i,c in enumerate(nb['cells']):
 if c['cell_type']!='code':continue
 s=''.join(c['source'])
 try:
  print('cell',i,flush=True)
  exec(compile(s,f'notebook cell {i}','exec'),ns)
 except Exception as e:
  print('CELL_ERROR',i,type(e).__name__,str(e),flush=True)
  raise
print('FINAL',ns['champion_name'],ns['test_metrics'][['Accuracy','Log loss','Draw recall']].to_dict('index'))
print('FORECASTS',ns['upcoming_champion_predictions'][['HomeTeam','AwayTeam','P_Home','P_Draw','P_Away','PredictionLabel']].to_string(index=False))
