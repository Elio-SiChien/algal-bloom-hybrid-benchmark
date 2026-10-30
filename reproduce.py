"""Reproduce the V3 paper in an isolated output directory; frozen inputs are read-only."""
from pathlib import Path
import argparse, datetime, hashlib, importlib.metadata, json, os, shutil, subprocess, sys, urllib.request, zipfile
ROOT=Path(__file__).resolve().parent
METHODS={'PM-GP':'M4r','NGR':'M3','T-PGNN':'M1','HAPP':'M2'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_bytes((json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n').encode())
def check_inputs():
 manifest=json.loads((ROOT/'data/integrity_manifest.json').read_text(encoding='utf8'))
 if not manifest:raise RuntimeError('Empty data manifest')
 for name,want in manifest.items():
  p=ROOT/name
  if not p.is_file() or sha(p)!=want:raise RuntimeError('Missing or changed input: '+name)
 for item in json.loads((ROOT/'reference/frozen_source_manifest.json').read_text(encoding='utf8')):
  if sha(ROOT/item['path'])!=item['sha256']:raise RuntimeError('Frozen algorithm changed: '+item['path'])
 return len(manifest)
def predictions(work,local=None):
 manifest=json.loads((ROOT/'release_manifest.json').read_text(encoding='utf8'))
 cache=ROOT/'.cache';cache.mkdir(exist_ok=True)
 asset=Path(local).resolve() if local else cache/manifest['asset']
 if not asset.exists():
  if local:raise FileNotFoundError(asset)
  temporary=asset.with_suffix('.download')
  try:
   with urllib.request.urlopen(manifest['url'],timeout=120) as response,temporary.open('wb') as out:shutil.copyfileobj(response,out)
  except Exception as e:raise RuntimeError('Cannot obtain fixed release predictions; no substitute data used') from e
  if sha(temporary)!=manifest['sha256']:raise RuntimeError('Downloaded prediction archive checksum failed')
  temporary.replace(asset)
 if sha(asset)!=manifest['sha256']:raise RuntimeError('Prediction archive checksum failed')
 dest=work/'reference_runs';dest.mkdir()
 with zipfile.ZipFile(asset) as z:
  if set(z.namelist())!=set(manifest['members']):raise RuntimeError('Unexpected archive members')
  for name,want in manifest['members'].items():
   target=(dest/name).resolve()
   if not target.is_relative_to(dest.resolve()):raise RuntimeError('Unsafe archive path')
   b=z.read(name)
   if hashlib.sha256(b).hexdigest()!=want:raise RuntimeError('Corrupt prediction member: '+name)
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
 return dest
def invoke(cmd,work,log,extra_env=None):
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONIOENCODING='utf8')
 env.update(extra_env or {})
 with log.open('wb') as out:
  result=subprocess.run(cmd,cwd=work,stdout=out,stderr=subprocess.STDOUT,env=env)
 if result.returncode:raise RuntimeError('Command failed; inspect '+str(log))
def configure(work):
 sys.path.insert(0,str(work/'experiments'))
 import figures_paper as f
 f.data.set_version('v3');f.SPLITS=('eval',);f.PRED_FILE='test_predictions.csv'
 f.SITES=[(k,v[3]) for k,v in f.v3.SITES.items()]
 f.SEGMENTS=(('train',f.v3.START,f.v3.TRAIN_YEAR_END),('eval',f.v3.EVAL_START,f.v3.END))
 f.EVAL_START=f.v3.EVAL_START;f.EVAL_LABEL='2024'
 by={x[1]:x for x in f.METHODS}
 f.METHODS=[(by[n][0],n,tag,by[n][3]) for n,tag in [('PM-GP','Residual correction'),('NGR','Inside'),('T-PGNN','Alongside'),('HAPP','Potential')]]
 return f
def compare_tables(actual,path,keys,tolerance=1e-10,rounded=False):
 import numpy as np,pandas as pd
 expected=pd.read_csv(path).set_index(keys).sort_index();got=actual.set_index(keys).sort_index()
 if not expected.index.equals(got.index):raise AssertionError('Row identities differ: '+path.name)
 for col in expected.columns:
  if col not in got:raise AssertionError('Missing column '+col)
  if pd.api.types.is_numeric_dtype(expected[col]):
   values=got[col].round(4) if rounded else got[col]
   np.testing.assert_allclose(values,expected[col],atol=tolerance,rtol=0,equal_nan=True,err_msg=path.name+': '+col)
  elif not expected[col].fillna('').equals(got[col].fillna('')):raise AssertionError('Column differs: '+col)
def results(args,work):
 import numpy as np,pandas as pd
 from scipy.stats import wilcoxon
 runs=predictions(work,args.predictions);f=configure(work)
 print('Recomputing scores and 14-calendar-day block-bootstrap intervals...',flush=True)
 tab,season,lead,store,missing,chance=f.collect(runs)
 if missing or len(tab)!=60:raise AssertionError('Incomplete method/waterbody coverage')
 out=work/'results';out.mkdir()
 tab.to_csv(out/'metrics_full_precision.csv',index=False)
 compare_tables(tab,ROOT/'reference/Table2_metrics.csv',['lake','period','method'],rounded=True)
 # Full-precision run scores are distinct from the four-decimal published table.
 raw_errors=[]
 for row in tab.itertuples():
  if row.method in METHODS:
   expected=json.loads((runs/f'{METHODS[row.method]}_{row.lake}/test_scores.json').read_text())['eval']['observed_main']
  else:
   key={'Persistence':'persistence','Climatology':'climatology'}[row.method]
   expected=json.loads((runs/f'references_{row.lake}/test_scores.json').read_text())['eval'][key]['observed_main']
  for col in ['mae','rmse','nse','hss','trend_accuracy','n']:
   error=abs(getattr(row,col)-expected[col]);raw_errors.append(error)
   if error>1e-10:raise AssertionError(f'Full precision mismatch: {row.lake}/{row.method}/{col}: {error}')
 tests=f.lake_level_tests(tab);tests.to_csv(out/'Table3_lake_level_tests.csv',index=False)
 compare_tables(tests,ROOT/'reference/Table3_lake_level_tests.csv',['method'],rounded=True)
 lead.to_csv(out/'lead_time_intervals.csv',index=False)
 compare_tables(lead,ROOT/'reference/lead_time_skill.csv',['lake','method','lead'],rounded=True)
 tab.round(4).to_csv(out/'Table2_metrics.csv',index=False)
 tab.groupby('method').median(numeric_only=True).to_csv(out/'Table2_medians.csv')
 threshold=[]
 for delta in [.01,.02,.03,.04]:
  f.evaluate.DELTA=delta;rs=[]
  for site,s in store.items():rs+=f.score_rows(site,'eval',s['obs'],s['allp'])[0]
  x=pd.DataFrame(rs)
  for method,g in x.groupby('method'):
   threshold.append(dict(delta=delta,method=method,hss_med=g.hss.median(),acc_med=g.trend_accuracy.median(),rise_med=g.rise_hit.median(),fall_med=g.fall_hit.median(),hss_pos=int((g.hss>0).sum()),p=float(wilcoxon(g.hss,alternative='greater').pvalue) if method in METHODS else np.nan))
 f.evaluate.DELTA=.02
 thresholds=pd.DataFrame(threshold);thresholds['rank_hss']=thresholds.groupby('delta').hss_med.rank(ascending=False)
 thresholds.to_csv(out/'TableS_trend_threshold_sensitivity.csv',index=False)
 compare_tables(thresholds,ROOT/'reference/TableS_trend_threshold_sensitivity.csv',['delta','method'],rounded=True)
 cases=[]
 for site,letters in [('fuxian','cdef'),('shahu','ghij')]:
  s=store[site]
  for panel,(label,k) in zip(letters,f.pick_cases(s['obs'],s['origins'])):
   truth=int(f.evaluate.trend_class(s['obs'][[k]])[0])
   for method in METHODS:
    p=np.where(np.isfinite(s['obs'][[k]]),s['allp'][method][[k]],np.nan)
    pred=int(f.evaluate.trend_class(p)[0])
    cases.append(dict(panel=panel,waterbody=site,origin=str(s['origins'][k].date()),case=label,method=method,observed_class=truth,predicted_class=pred,correct=truth==pred,observed_slope_ln_per_day=float(f.evaluate.theil_sen(s['obs'][[k]]*f.evaluate.LN10)[0]),predicted_slope_ln_per_day=float(f.evaluate.theil_sen(p*f.evaluate.LN10)[0]),observed_target_days=int(np.isfinite(s['obs'][k]).sum())))
 cases=pd.DataFrame(cases);cases.to_csv(out/'case_windows.csv',index=False)
 compare_tables(cases,ROOT/'reference/case_windows.csv',['panel','method'])
 nutrients=pd.read_csv(ROOT/'reference/TableS1_waterbodies.csv')
 for i,row in nutrients.iterrows():
  site=row.Waterbody.replace('Lake ','').replace(' Reservoir','').lower();d=store[site]['daily']
  for var,col in [('tn','Observed_TN_median_mg_L'),('tp','Observed_TP_median_mg_L')]:
   value=d.loc[~d[var+'_filled']&d[var].notna(),var].median()
   if abs(value-row[col])>1e-10:raise AssertionError('Nutrient mismatch')
   nutrients.loc[i,col]=value
 nutrients.to_csv(out/'TableS1_waterbodies.csv',index=False)
 for name in ['fig5','fig3_v3','fig7_radar_v3','fig8_days_ahead_v3','fig4_rises_falls_v3','fig6_ranking_v3']:
  print('Drawing',name,flush=True)
  call=getattr(f,name)
  if name=='fig5':call(store,tab,out)
  elif name=='fig3_v3':call(store,out)
  elif name=='fig8_days_ahead_v3':call(lead,out)
  elif name=='fig4_rises_falls_v3':call(tab,store,chance,out)
  else:call(tab,out)
 invoke([sys.executable,'-X','utf8','tools/plot_v3_si.py'],work,work/'quality_control_figures.log')
 invoke([sys.executable,'-X','utf8','tools/station_map.py','--table',str(ROOT/'reference/TableS1_waterbodies.csv'),'--out',str(out)],work,work/'station_map.log')
 shutil.copytree(ROOT/'assets',out/'explanatory_assets')
 h=tab[tab.method.isin(METHODS)].pivot(index='lake',columns='method',values='hss')
 best=h.idxmax(axis=1).value_counts().to_dict()
 assert best=={'NGR':6,'T-PGNN':2,'PM-GP':2}
 assert (h.NGR>0).sum()==10 and (h['T-PGNN']>0).sum()==10
 report={'results_recomputed':True,'full_precision_score_comparisons':len(raw_errors),'max_absolute_score_error':max(raw_errors),'score_tolerance':1e-10,'published_table_comparison':'Compare at the original four-decimal reporting precision','hss_first_counts':best,'hss_positive_counts':(h>0).sum().to_dict(),'threshold_descriptions_verified':True,'threshold_p_values_reproduced':True,'case_direction_counts':cases.groupby('method').correct.sum().to_dict(),'full_retraining_verified':False,'figures_are_numeric_reproductions_not_pixel_identical':True}
 dump(work/'verification.json',report);print(json.dumps(report,indent=2),flush=True)
def train_or_smoke(args,work):
 import pandas as pd
 f=configure(work);sites=list(f.v3.SITES) if args.site=='all' else [args.site]
 if args.mode=='smoke' and args.site=='all':sites=['chaohu']
 methods=list(METHODS.values()) if args.method=='all' else [METHODS.get(args.method,args.method)]
 if not set(sites)<=set(f.v3.SITES) or not set(methods)<=set(METHODS.values()):raise ValueError('Unknown site/method')
 if args.mode=='train' and 'M1' in methods:
  import torch
  if os.environ.get('M1_DEVICE','cuda').startswith('cuda') and not torch.cuda.is_available():
   raise RuntimeError('Formal T-PGNN defaults to NVIDIA CUDA. Install a matching CUDA PyTorch build; smoke mode explicitly uses the original CPU test switch.')
 checks=[]
 for site in sites:
  for method in methods:
   print(args.mode,method,site,flush=True)
   cmd=[sys.executable,'-X','utf8','experiments/run_method.py','--method',method,'--site',site,'--data','v3','--dry-run' if args.mode=='smoke' else '--formal']
   invoke(cmd,work,work/f'{method}_{site}.log',{'M1_DEVICE':'cpu'} if args.mode=='smoke' else None)
   base=work/'results'/('dry_runs_v3' if args.mode=='smoke' else 'runs_v3')/f'{method}_{site}'
   p=pd.read_csv(base/'test_predictions.csv');final=p[p.series=='final']
   if set(final.lead)!=set(range(1,8)) or not final.pred_log10.notna().all():raise AssertionError('Incomplete seven-day forecast')
   if args.mode=='smoke':
    if not final.origin.str.startswith('2023-').all():raise AssertionError('Smoke test used formal evaluation year')
   checks.append({'method':method,'site':site,'rows':len(final),'all_seven_days':True,'scope':'training-only pseudo-test' if args.mode=='smoke' else 'formal rerun'})
  if args.mode=='train':invoke([sys.executable,'-X','utf8','experiments/run_method.py','--method','references','--site',site,'--data','v3','--formal'],work,work/f'references_{site}.log')
 dump(work/'verification.json',{'mode':args.mode,'checks':checks,'full_formal_training_completed':args.mode=='train' and len(sites)==10 and len(methods)==4,'full_formal_retraining_verified':False,'note':'A completed training run still requires comparison with frozen predictions before claiming numerical replication.'})
def prepare(work):
 import numpy as np,pandas as pd
 data=work/'data/study_lakes/v3';data.rename(data.with_name('frozen_v3'))
 raw=pd.concat([pd.read_csv(p) for p in (work/'data/study_lakes/frozen_v3/intermediate').glob('readings_4h_raw_*.csv.gz')],ignore_index=True)
 raw.to_csv(work/'ten_station_subset.csv.gz',index=False)
 invoke([sys.executable,'-X','utf8','experiments/build_v3.py','--national',str(work/'ten_station_subset.csv.gz'),'--lock'],work,work/'prepare.log')
 for p in data.glob('qc_daily_v3_*.csv'):
  original=pd.read_csv(work/'data/study_lakes/frozen_v3'/p.name);new=pd.read_csv(p)
  pd.testing.assert_frame_equal(original,new,check_exact=False,atol=1e-10,rtol=0)
 pd.testing.assert_frame_equal(pd.read_csv(data/'origins_v3.csv'),pd.read_csv(work/'data/study_lakes/frozen_v3/origins_v3.csv'))
 dump(work/'verification.json',{'daily_data_and_origins_rebuilt':True,'source':'Ten published station subsets, not the original national archive','tolerance':1e-10})
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--mode',choices=['results','smoke','train','prepare'],default='results')
 ap.add_argument('--method',default='all');ap.add_argument('--site',default='all')
 ap.add_argument('--out',type=Path);ap.add_argument('--predictions',help='Optional local copy of the checksum-verified release ZIP')
 args=ap.parse_args();n=check_inputs()
 work=(args.out or ROOT/'outputs'/f'{args.mode}_{datetime.datetime.now():%Y%m%d_%H%M%S}').resolve()
 if work.exists():raise FileExistsError('Choose a new output directory: '+str(work))
 # Never create an output directory inside frozen inputs or algorithm sources.
 for parent in [ROOT/'data',ROOT/'reference',ROOT/'experiments',ROOT/'assets',ROOT/'tools',ROOT/'.git',ROOT/'.cache']:
  if work.is_relative_to(parent.resolve()):raise ValueError('Output must be outside frozen inputs')
 work.mkdir(parents=True)
 for name in ['experiments','data','tools']:shutil.copytree(ROOT/name,work/name,ignore=shutil.ignore_patterns('__pycache__'))
 versions={x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','matplotlib']}
 dump(work/'environment.json',{'python':sys.version,'packages':versions,'input_files_verified':n})
 if args.mode=='results':results(args,work)
 elif args.mode=='prepare':prepare(work)
 else:train_or_smoke(args,work)
 check_inputs();print('Completed:',work,flush=True)
if __name__=='__main__':main()
