"""Stream selected BTS columns from monthly ZIPs into compact Parquet."""
from pathlib import Path
from datetime import date
import csv,hashlib,json,zipfile
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['FlightDate','IATA_Code_Operating_Airline','DOT_ID_Operating_Airline','Tail_Number','Origin','Dest',
        'CRSDepTime','DepTime','CRSElapsedTime','ActualElapsedTime','ArrDelay','ArrDelayMinutes','ArrDel15',
        'Cancelled','Diverted','Distance','Duplicate']
SCHEMA=pa.schema([('flight_id',pa.string()),('flight_date',pa.date32()),('carrier',pa.string()),('tail_hash',pa.string()),
('origin',pa.string()),('destination',pa.string()),('crs_dep_time',pa.int16()),('actual_dep_time',pa.int16()),
('crs_elapsed_min',pa.float32()),('actual_elapsed_min',pa.float32()),('arrival_delay_min',pa.float32()),
('arrival_delay_minutes',pa.float32()),('arrival_delay_15',pa.int8()),('cancelled',pa.int8()),('diverted',pa.int8()),
('distance_mi',pa.float32()),('duplicate_flag',pa.bool_()),('source_month',pa.string())])

def number(value,cast=float):
    if value is None or not str(value).strip():return None
    try:
        x=float(value)
        if not (x==x and abs(x)<1e30):return None
        return cast(x)
    except (ValueError,TypeError,OverflowError):return None

def main():
    (ROOT/'data/cache').mkdir(parents=True,exist_ok=True)
    output=ROOT/'data/cache/bts_flights.parquet';tmp=output.with_suffix('.part.parquet')
    writer=None;total=duplicates=missing_label=0;batch=[]
    try:
        for month in range(1,7):
            path=next((ROOT/'data/raw').glob(f'*_2025_{month}.zip'),None)
            if path is None:raise FileNotFoundError(f'Missing 2025-{month:02d} archive; run scripts/download_data.py')
            month_label=f'2025-{month:02d}'
            with zipfile.ZipFile(path) as archive:
                csv_members=[n for n in archive.namelist() if n.lower().endswith('.csv')]
                if len(csv_members)!=1:raise ValueError(f'Expected one CSV in {path.name}, found {len(csv_members)}')
                import io
                with archive.open(csv_members[0]) as binary:
                    text=io.TextIOWrapper(binary,encoding='utf-8-sig',newline='')
                    reader=csv.DictReader(text);reader.fieldnames=[x.strip() for x in reader.fieldnames]
                    absent=sorted(set(FIELDS)-set(reader.fieldnames))
                    if absent:raise ValueError(f'{month_label}: BTS file lacks selected fields {absent}')
                    for row_id,row in enumerate(reader,1):
                        row={k.strip():v for k,v in row.items() if k is not None}
                        dup=str(row.get('Duplicate','')).strip().upper()=='Y'
                        if dup:duplicates+=1;continue
                        flight_date=date.fromisoformat(row['FlightDate'].strip())
                        tail=str(row.get('Tail_Number') or '').strip().upper()
                        carrier=str(row.get('IATA_Code_Operating_Airline') or row.get('DOT_ID_Operating_Airline') or '').strip()
                        tail_hash=hashlib.sha256(('aircraft-v1|'+tail).encode()).hexdigest()[:20] if tail else None
                        delay15=number(row.get('ArrDel15'),int)
                        canceled=number(row.get('Cancelled'),int) or 0;diverted=number(row.get('Diverted'),int) or 0
                        if canceled==0 and diverted==0 and delay15 is None:missing_label+=1
                        batch.append({'flight_id':f'{month_label}-{row_id:09d}','flight_date':flight_date,'carrier':carrier or None,
                            'tail_hash':tail_hash,'origin':(row.get('Origin') or '').strip() or None,
                            'destination':(row.get('Dest') or '').strip() or None,
                            'crs_dep_time':number(row.get('CRSDepTime'),int),'actual_dep_time':number(row.get('DepTime'),int),
                            'crs_elapsed_min':number(row.get('CRSElapsedTime')),'actual_elapsed_min':number(row.get('ActualElapsedTime')),
                            'arrival_delay_min':number(row.get('ArrDelay')),'arrival_delay_minutes':number(row.get('ArrDelayMinutes')),
                            'arrival_delay_15':delay15,'cancelled':canceled,'diverted':diverted,
                            'distance_mi':number(row.get('Distance')),'duplicate_flag':dup,'source_month':month_label})
                        total+=1
                        if len(batch)>=50000:
                            table=pa.Table.from_pylist(batch,schema=SCHEMA)
                            if writer is None:writer=pq.ParquetWriter(tmp,SCHEMA,compression='zstd',use_dictionary=True)
                            writer.write_table(table);batch.clear()
            print(f'parsed {month_label}; retained rows={total:,}',flush=True)
        if batch:
            table=pa.Table.from_pylist(batch,schema=SCHEMA)
            if writer is None:writer=pq.ParquetWriter(tmp,SCHEMA,compression='zstd',use_dictionary=True)
            writer.write_table(table);batch.clear()
        if writer is None:raise ValueError('No flight records were read')
        writer.close();writer=None;tmp.replace(output)
    finally:
        if writer is not None:writer.close()
        if tmp.exists():tmp.unlink()
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    manifest['prepared_data']={'path':'data/cache/bts_flights.parquet','rows':total,'duplicate_flagged_rows_excluded':duplicates,
        'rows_missing_completed_arrival_label':missing_label,'columns':SCHEMA.names,'aircraft_tail_numbers_hashed':True,
        'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'bytes':output.stat().st_size}
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest['prepared_data'],indent=2))

if __name__=='__main__':main()
