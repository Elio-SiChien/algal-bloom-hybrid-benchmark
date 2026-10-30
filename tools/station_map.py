"""Plot the ten station coordinates; manuscript cartographic SVG is a separate asset."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
def draw(table,out):
 d=pd.read_csv(table);fig,ax=plt.subplots(figsize=(8,7))
 ax.scatter(d.Longitude_E,d.Latitude_N,color='#26758a',s=25)
 for i,r in d.iterrows():ax.annotate(r.Waterbody,(r.Longitude_E,r.Latitude_N),xytext=(5,6+12*(i%2)),textcoords='offset points',fontsize=8)
 ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',title='Benchmark monitoring stations (coordinate plot)')
 ax.grid(alpha=.15);fig.tight_layout();fig.savefig(Path(out)/'Fig2_station_coordinates.pdf');plt.close(fig)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--table',required=True);p.add_argument('--out',required=True);a=p.parse_args();Path(a.out).mkdir(exist_ok=True,parents=True);draw(a.table,a.out)
