"""Collect final diagnostic metrics without mixing snapshot families or test assumptions."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
rows=[]
for suffix in ['a','b','c','e','g','h']:
 folder=root/f'boundary_test_20261021_{suffix}'
 if not (folder/'events.jsonl').exists():continue
 events=[json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()]
 config=json.loads((folder/'config.json').read_text())
 for arm in config['arms'].split(','):
  matches=[e for e in events if e.get('arm')==arm and e['event'] in ('train_done','train')]
  if not matches:continue
  last=matches[-1]
  row={k:last[k] for k in ('arm','epoch','seconds','missing_with_any_cos99','missing_with_any_inf02','missing_with_4_accurate_inf02','missing_with_4_agree_inf02','pinned_max_error')}
  row['missing_with_5_accurate_inf02']=sum(n>=5 for n in last['accurate_member_counts_inf02'])
  row.update({'run':folder.name,'complete':last['event']=='train_done','best_training_L1':min(last['loss_by_member']),'best_cosine_min_across_missing':min(last['best_cosine_by_missing_row']),'best_inf_error_max_across_missing':max(last['best_inf_error_by_missing_row']),'training_loss':'MSE' if 'mse' in arm else 'L1','F6_head':'teacher exact, fixed' if 'fixedtail' in arm else 'random, trained','data':config.get('dataset_from') or folder.name})
  rows.append(row)
(root/'committee_test_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(r['run'],r['arm'],r['epoch'],'complete',r['complete'],'L1',round(r['best_training_L1'],7),'any/4accurate',r['missing_with_any_inf02'],r['missing_with_4_accurate_inf02'])
