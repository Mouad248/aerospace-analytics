from __future__ import annotations
import json, math
from datetime import datetime, timezone
from pathlib import Path
import duckdb, joblib, lightgbm as lgb, networkx as nx, numpy as np, pandas as pd
from airportsdata import load as load_airports
from ortools.sat.python import cp_model
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.calibration import calibration_curve
from zoneinfo import ZoneInfo

CATS=['carrier','origin','destination','route','dep_hour','day_of_week','month']
NUMS=['distance_mi','origin_28d_rate','destination_28d_rate','route_28d_rate','carrier_28d_rate']
FEATURES=CATS+NUMS

def make_features(root:Path)->Path:
    src=str(root/'data/cache/bts_flights.parquet').replace("'","''")
    if not (root/'data/cache/bts_flights.parquet').exists(): raise FileNotFoundError('Run scripts/prepare_data.py first.')
    out=root/'data/cache/model_features.parquet'; tmp=out.with_suffix('.part.parquet')
    con=duckdb.connect()
    # Rolling outcomes are strictly from dates before the flight date. Flight-level values
    # are averaged within each date/partition before the preceding 28-day range window.
    con.execute(f"""COPY (
      WITH base AS (
        SELECT *, (cancelled=1 OR diverted=1 OR arrival_delay_15=1)::INTEGER AS disrupted,
          CASE WHEN cancelled=0 AND diverted=0 THEN arrival_delay_min ELSE NULL END AS delay_target,
          coalesce(carrier,'UNK') || '|' || coalesce(origin,'UNK') || '|' || coalesce(destination,'UNK') AS route,
          CAST(floor(coalesce(crs_dep_time,0)/100) AS INTEGER) AS dep_hour,
          dayofweek(flight_date)::INTEGER AS day_of_week, month(flight_date)::INTEGER AS month
        FROM read_parquet('{src}')
      ), daily AS (
        SELECT flight_date, origin, destination, carrier, route,
          avg(disrupted) AS origin_rate, avg(disrupted) AS destination_rate,
          avg(disrupted) AS route_rate, avg(disrupted) AS carrier_rate
        FROM base GROUP BY ALL
      ), history AS (
        SELECT *,
          avg(origin_rate) OVER (PARTITION BY origin ORDER BY flight_date RANGE BETWEEN INTERVAL 28 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING) AS origin_28d_rate,
          avg(destination_rate) OVER (PARTITION BY destination ORDER BY flight_date RANGE BETWEEN INTERVAL 28 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING) AS destination_28d_rate,
          avg(route_rate) OVER (PARTITION BY route ORDER BY flight_date RANGE BETWEEN INTERVAL 28 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING) AS route_28d_rate,
          avg(carrier_rate) OVER (PARTITION BY carrier ORDER BY flight_date RANGE BETWEEN INTERVAL 28 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING) AS carrier_28d_rate
        FROM daily
      )
      SELECT b.flight_id,b.flight_date,b.carrier,b.origin,b.destination,b.route,b.dep_hour,b.day_of_week,b.month,
        b.distance_mi,b.disrupted,b.delay_target,b.cancelled,b.diverted,b.arrival_delay_15,b.tail_hash,
        h.origin_28d_rate,h.destination_28d_rate,h.route_28d_rate,h.carrier_28d_rate,
        b.crs_dep_time,b.actual_dep_time,b.crs_elapsed_min,b.actual_elapsed_min,b.arrival_delay_min
      FROM base b JOIN history h USING(flight_date,origin,destination,carrier,route)
    ) TO '{str(tmp).replace("'","''")}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    con.close(); tmp.replace(out)
    return out

def load_features(path:Path)->pd.DataFrame:
    con=duckdb.connect(); df=con.execute(f"SELECT * FROM read_parquet('{str(path).replace(chr(39), chr(39)*2)}')").df(); con.close()
    df['flight_date']=pd.to_datetime(df.flight_date)
    for c in CATS: df[c]=df[c].fillna('UNK').astype('category')
    return df

def _calibration_rows(y,p):
    obs,pred=calibration_curve(y,p,n_bins=10,strategy='quantile')
    return [{'predicted_probability':float(x),'observed_rate':float(yv)} for x,yv in zip(pred,obs)]

def train(root:Path):
    feat_path=make_features(root); df=load_features(feat_path)
    train=df[(df.flight_date<'2025-05-01') & df.disrupted.notna()].copy()
    valid=df[(df.flight_date>='2025-05-01')&(df.flight_date<'2025-06-01')&df.disrupted.notna()].copy()
    test=df[(df.flight_date>='2025-06-01')&(df.flight_date<'2025-07-01')&df.disrupted.notna()].copy()
    # A deterministic cap keeps the demo reproducible on a laptop while retaining a large sample.
    if len(train)>650000:
        train=train.sample(n=650000,random_state=2026)
    clf=lgb.LGBMClassifier(n_estimators=450,learning_rate=.045,num_leaves=31,max_depth=-1,
        subsample=.85,colsample_bytree=.85,reg_lambda=2,class_weight='balanced',random_state=2026,
        n_jobs=4,verbosity=-1)
    clf.fit(train[FEATURES],train.disrupted.astype(int),categorical_feature=CATS)
    raw_valid=clf.predict_proba(valid[FEATURES])[:,1]
    calibrator=IsotonicRegression(out_of_bounds='clip').fit(raw_valid,valid.disrupted.astype(float))
    p=calibrator.predict(clf.predict_proba(test[FEATURES])[:,1])
    y=test.disrupted.astype(int).to_numpy()
    # Quantile regression is restricted to completed, non-diverted observations.
    qtrain=train[train.delay_target.notna()]
    if len(qtrain)>450000:qtrain=qtrain.sample(n=450000,random_state=2027)
    qvalid=valid[valid.delay_target.notna()]; qtest=test[test.delay_target.notna()]
    quant=lgb.LGBMRegressor(objective='quantile',alpha=.9,n_estimators=500,learning_rate=.04,
       num_leaves=31,subsample=.85,colsample_bytree=.85,reg_lambda=2,random_state=2027,n_jobs=4,verbosity=-1)
    quant.fit(qtrain[FEATURES],qtrain.delay_target,categorical_feature=CATS)
    # Marginal split-conformal correction: fit only on the May calibration month,
    # then apply the frozen offset to June. The finite-sample order statistic
    # targets 90% coverage under exchangeability; temporal drift can reduce it.
    qvalid_pred=quant.predict(qvalid[FEATURES])
    qvalid_y=qvalid.delay_target.to_numpy()
    residuals=qvalid_y-qvalid_pred
    conformal_level=min(1.0,math.ceil((len(residuals)+1)*.9)/len(residuals))
    q90_offset=float(np.quantile(residuals,conformal_level,method='higher'))
    qpred_raw=quant.predict(qtest[FEATURES])
    qpred=qpred_raw+q90_offset
    # Baseline is the preceding 28-day route disruption rate; use training prevalence only if history absent.
    base=test.route_28d_rate.fillna(float(train.disrupted.mean())).clip(0,1).to_numpy()
    briers=[]
    for name,prob in [('lightgbm_isotonic',p),('route_28d_baseline',base)]:
        briers.append((name,float(brier_score_loss(y,prob))))
    def safe_auc(fn):
        try:return float(fn(y,p))
        except ValueError:return None
    qy=qtest.delay_target.to_numpy()
    results={'sample':{'prepared_rows':len(df),'train_rows_used':len(train),'validation_rows':len(valid),'june_test_rows':len(test),
        'june_completed_nondiverted_delay_rows':len(qtest),'test_positive_rate':float(np.mean(y))},
      'disruption':{'roc_auc':safe_auc(roc_auc_score),'average_precision':float(average_precision_score(y,p)),
        'brier_score':dict(briers)['lightgbm_isotonic'],'baseline_route_28d_brier':dict(briers)['route_28d_baseline'],
        'calibration':_calibration_rows(y,p)},
      'delay_q90':{'mae':float(mean_absolute_error(qy,qpred)),'raw_mae':float(mean_absolute_error(qy,qpred_raw)),
        'p90_prediction':float(np.quantile(qpred,.9)),'raw_p90_prediction':float(np.quantile(qpred_raw,.9)),
        'empirical_coverage':float(np.mean(qy<=qpred)),'raw_empirical_coverage':float(np.mean(qy<=qpred_raw)),
        'validation_coverage_before_calibration':float(np.mean(qvalid_y<=qvalid_pred)),
        'validation_coverage_after_calibration':float(np.mean(qvalid_y<=qvalid_pred+q90_offset)),
        'calibration_method':'May-only finite-sample split-conformal additive correction',
        'calibration_rows':int(len(qvalid)),'calibration_quantile_level':float(conformal_level),
        'calibration_offset_min':q90_offset,'actual_p90':float(np.quantile(qy,.9)),'rows':len(qtest)},
      'features':FEATURES,'validation_month':'2025-05','test_month':'2025-06',
      'limits':['Only pre-departure schedule/operator/location/distance and prior observed 28-day history are model inputs.',
        'Six months of U.S. records; June is one temporal holdout month, and results are not transferable without validation.',
        'No weather forecast, passenger connections, crew, gate, maintenance or airline-cost data.',
        'CP-SAT buffer allocation is an assumption-led scenario, not measured airline recovery benefit.']}
    # June predictions retained without aircraft tail hashes or raw IDs.
    pred=pd.DataFrame({'flight_date':test.flight_date.dt.date.astype(str),'carrier':test.carrier.astype(str).to_numpy(),
       'origin':test.origin.astype(str).to_numpy(),'destination':test.destination.astype(str).to_numpy(),
       'disruption_probability':p})
    allq=pd.DataFrame({'flight_date':qtest.flight_date.dt.date.astype(str),'carrier':qtest.carrier.astype(str).to_numpy(),
       'origin':qtest.origin.astype(str).to_numpy(),'destination':qtest.destination.astype(str).to_numpy(),
       'disruption_probability':calibrator.predict(clf.predict_proba(qtest[FEATURES])[:,1]),'q90_arrival_delay_min':qpred,
       'tail_hash':qtest.tail_hash.fillna('').astype(str).to_numpy(),'scheduled_dep':qtest.crs_dep_time.to_numpy(),
       'actual_dep':qtest.actual_dep_time.to_numpy(),'scheduled_block':qtest.crs_elapsed_min.to_numpy(),
       'actual_block':qtest.actual_elapsed_min.to_numpy(),'arrival_delay_min':qtest.arrival_delay_min.to_numpy()})
    root.joinpath('models').mkdir(exist_ok=True);root.joinpath('reports/latest').mkdir(parents=True,exist_ok=True)
    joblib.dump({'disruption':clf,'calibrator':calibrator,'delay_q90':quant,'delay_q90_offset_min':q90_offset,
                 'features':FEATURES,'categoricals':CATS},root/'models/airline_models.joblib',compress=3)
    (root/'reports/latest/results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    pred.to_csv(root/'reports/latest/june_disruption_predictions.csv',index=False)
    allq.to_parquet(root/'data/cache/june_q90_predictions.parquet',index=False)
    print(json.dumps(results,indent=2)); return results

def rotation_and_optimization(root:Path,budget_min=240):
    f=root/'data/cache/june_q90_predictions.parquet'
    if not f.exists():raise FileNotFoundError('Run airline training first.')
    df=pd.read_parquet(f); day=df.flight_date.max(); day_rows=df[df.flight_date==day].copy()
    # Rotation reconstruction is for the selected held-out day only.
    rotation_rows=day_rows[day_rows.tail_hash.ne('')]
    airports=load_airports('IATA')
    def utc_time(row,actual=False,elapsed=0):
        code=row.origin; airport=airports.get(code)
        if not airport:return None
        raw=row.actual_dep if actual and pd.notna(row.actual_dep) else row.scheduled_dep
        if pd.isna(raw):return None
        hh=int(raw)//100;mm=int(raw)%100
        try:
            day=pd.Timestamp(row.flight_date).date()
            if hh==24 and mm==0:day=day+pd.Timedelta(days=1);hh=0
            local=datetime.combine(day,datetime.min.time()).replace(hour=hh,minute=mm,tzinfo=ZoneInfo(airport['tz']))
            return local.astimezone(timezone.utc)+pd.Timedelta(minutes=elapsed)
        except Exception:return None
    # Tail hashes are used only for same-day sequence construction, never exported.
    graph=nx.DiGraph(); rows=[]
    for idx,r in rotation_rows.iterrows():
        elapsed=float(r.actual_block) if pd.notna(r.actual_block) else (float(r.scheduled_block) if pd.notna(r.scheduled_block) else 0.0)
        dep=utc_time(r,True); arr=utc_time(r,True,elapsed)
        if dep is None or arr is None:continue
        rows.append((r.tail_hash,dep,arr,r.origin,r.destination,float(r.arrival_delay_min or 0),idx))
    rows.sort(key=lambda z:(z[0],z[1]))
    links=[]
    for a,b in zip(rows,rows[1:]):
        if a[0]==b[0] and a[3+1]==b[3] and 25 <= (b[1]-a[2]).total_seconds()/60 <= 720:
            links.append((a,b))
    for i,(a,b) in enumerate(links):
        graph.add_edge(i,i+1,turn_minutes=(b[1]-a[2]).total_seconds()/60,prior_arrival_delay=a[5])
    next_delays=[b[5] for a,b in links];prior_delays=[a[5] for a,b in links]
    corr=float(np.corrcoef(prior_delays,next_delays)[0,1]) if len(links)>1 and np.std(prior_delays)>0 and np.std(next_delays)>0 else None
    # Scenario candidates are top June-day q90 risks; allocate 5-minute chunks, max 30 per flight.
    cand=day_rows.sort_values('q90_arrival_delay_min',ascending=False).head(30).reset_index(drop=True)
    model=cp_model.CpModel(); units=[model.NewIntVar(0,6,f'b{i}') for i in range(len(cand))]
    model.Add(sum(units)<=budget_min//5); objective=[]
    for i,r in cand.iterrows():
        coeff=max(0,int(round(float(r.disruption_probability)*float(r.q90_arrival_delay_min)*10)))
        objective.append(coeff*units[i])
    # Maximize the authored risk-weighted buffer score under the shared budget.
    model.Maximize(sum(objective));solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=5
    status=solver.Solve(model); allocation=[solver.Value(x)*5 for x in units] if status in (cp_model.OPTIMAL,cp_model.FEASIBLE) else [0]*len(cand)
    cand['scenario_buffer_min']=allocation;cand['residual_q90_min']=(cand.q90_arrival_delay_min-cand.scenario_buffer_min).clip(lower=0)
    # Export only a compact pseudonymous tail rotation summary and aggregate statistics.
    rotations={'heldout_day':str(day),'tail_rows_with_timezone':len(rows),'plausible_same_tail_airport_turn_links':len(links),
      'prior_arrival_vs_next_arrival_delay_correlation_descriptive':corr,
      'scenario':{'candidate_flights':len(cand),'assumed_total_buffer_budget_min':budget_min,'allocated_min':int(sum(allocation)),
        'mean_q90_before_min':float(cand.q90_arrival_delay_min.mean()),'mean_q90_after_min':float(cand.residual_q90_min.mean()),
        'interpretation':'Deterministic what-if allocation with an assumed objective; not an observed schedule intervention.'}}
    (root/'reports/latest/operations_scenario.json').write_text(json.dumps(rotations,indent=2),encoding='utf-8')
    cand[['flight_date','carrier','origin','destination','q90_arrival_delay_min','scenario_buffer_min','residual_q90_min']].to_csv(root/'reports/latest/buffer_scenario.csv',index=False)
    return rotations
