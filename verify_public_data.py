"""Read-only integrity and stored-outcome checks; never simulates or resamples."""
from pathlib import Path
from collections import defaultdict
import csv,gzip,hashlib,json

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'public_data/p4_v8810_20261024'

def main():
    manifest=DATA/'SOURCE_DATA_MANIFEST.csv'
    with manifest.open(encoding='utf-8',newline='') as f:entries=list(csv.DictReader(f))
    for row in entries:
        path=(ROOT/row['public_path']).resolve()
        if not path.is_relative_to(DATA.resolve()):raise ValueError('Manifest path escapes the data directory.')
        raw=path.read_bytes()
        if len(raw)!=int(row['bytes']) or hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('Data identity mismatch: '+row['public_path'])

    groups=defaultdict(lambda:dict(rows=0,roots=set(),decisions=0,authorizations=0,refreshes=0,under_target_authorizations=0,missions_with_under_target_authorization=0))
    path=DATA/'v8_7/results/test/adaptive_episodes.csv.gz'
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            key=(row['policy'],row['true_hazard']);g=groups[key]
            root=row['root_seed']  # Retain exact64-bit identity as text.
            if root in g['roots']:raise ValueError('Duplicate policy/mission root within hazard stratum.')
            g['rows']+=1;g['roots'].add(root)
            for field,source in [('decisions','decisions'),('authorizations','final_no_refresh'),('refreshes','refreshes'),('under_target_authorizations','authorized_under_target'),('missions_with_under_target_authorization','any_authorized_under_target')]:g[field]+=int(row[source])
    by_policy=defaultdict(lambda:dict(rows=0,roots=set(),decisions=0,authorizations=0,refreshes=0))
    for (policy,hazard),g in groups.items():
        b=by_policy[policy]
        if b['roots'] & g['roots']:raise ValueError('Root reused across hazard strata within a policy.')
        b['roots']|=g['roots']
        for field in ['rows','decisions','authorizations','refreshes']:b[field]+=g[field]
    if len(by_policy)!=4 or any(len(g['roots'])!=60000 for g in by_policy.values()):raise ValueError('Unexpected held-out policy/mission coverage.')
    first=next(iter(by_policy.values()))['roots']
    if any(g['roots']!=first for g in by_policy.values()):raise ValueError('Policy rows do not share the expected root-mission set.')
    primary=by_policy['posterior_mean+screen']
    if (primary['decisions'],primary['authorizations'],primary['refreshes'])!=(660000,189301,470699):raise ValueError('Primary stored totals differ from the frozen results.')
    high=groups[('posterior_mean+screen','0.012')]
    if (high['authorizations'],high['under_target_authorizations'])!=(33,33):raise ValueError('High-hazard negative evidence differs.')
    for g in by_policy.values():
        g['root_missions']=len(g.pop('roots'))
        g['authorization_fraction']=g['authorizations']/g['decisions']
        g['mean_refreshes']=g['refreshes']/g['root_missions']
    with (DATA/'v8_8/results/test_locked_v1_finalized_hotfix1/PRIMARY_GATE_TABLE.csv').open(encoding='utf-8',newline='') as f:gates=list(csv.DictReader(f))
    if len(gates)!=6 or sum(int(r['pass']) for r in gates)!=5:raise ValueError('Independent-spending negative result differs.')
    failed=[r['gate'] for r in gates if not int(r['pass'])]
    if failed!=['refresh_gain_vs_equal_spending']:raise ValueError('Unexpected failed spending gate.')
    print(json.dumps({'status':'PASS','data_files_hash_verified':len(entries),'v87_held_out_root_missions':60000,'v87_policy_mission_rows':sum(g['rows'] for g in by_policy.values()),'v87_policy_totals':dict(by_policy),'high_hazard_authorizations_under_target':'33/33','v88_frozen_gate_passes':'5/6','v88_failed_gate':failed[0],'scientific_runs':False,'bootstrap_runs':False,'physical_validation_claimed':False},indent=2))

if __name__=='__main__':main()
