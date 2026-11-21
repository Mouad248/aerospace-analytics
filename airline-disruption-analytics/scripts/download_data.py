"""Download pinned BTS 2025 monthly on-time ZIP archives."""
from pathlib import Path
import hashlib,json,os,tempfile,urllib.request

ROOT=Path(__file__).resolve().parents[1]
BASE='https://transtats.bts.gov/PREZIP/On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_2025_'
HASHES={1:'0feaabdbc9e4bd851ef717f342678cdcc5ea0822dd706359aab030bb1a5d1c24',
2:'080ad48e2826a6b4c3f39690f75a08dea9ad4e0831ff82958656cdb61693dd15',
3:'ad3f46a598bd529588bfcc004731b893b27999a681804a07a3841b2904d66d48',
4:'88b88ae1a9b49a03cb82c835481ac8dcf86ff2d378614655638889866767608f',
5:'261fb75b956abedca34a27db3ae7221437e1a2ce349b5ee9a49ec267d7555b29',
6:'0cb4038a49962ab92cba33f9840a8eec9671ef19a5afeedfa27998984476b609'}

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True);entries=[]
    for month,expected in HASHES.items():
        name=f'On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_2025_{month}.zip'
        dest=raw/name;url=BASE+str(month)+'.zip'
        if dest.exists() and digest(dest)!=expected:
            raise ValueError(f'{dest} exists but is not the pinned BTS file; preserve and inspect it manually')
        if not dest.exists():
            fd,tmp=tempfile.mkstemp(prefix=f'bts-2025-{month}-',suffix='.part',dir=raw);os.close(fd)
            h=hashlib.sha256()
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'airline-disruption-analytics/1.0'})
                with urllib.request.urlopen(request,timeout=180) as response,open(tmp,'wb') as out:
                    while block:=response.read(1024*1024):out.write(block);h.update(block)
                if h.hexdigest()!=expected:raise ValueError(f'{name} hash differs; source changed or transfer was incomplete')
                os.replace(tmp,dest)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
        entries.append({'month':f'2025-{month:02d}','url':url,'file':f'data/raw/{name}',
                        'bytes':dest.stat().st_size,'sha256':expected})
        print(f"verified {name} ({dest.stat().st_size:,} bytes)",flush=True)
    manifest={'publisher':'U.S. Bureau of Transportation Statistics, TranStats',
        'dataset':'Marketing Carrier On-Time Performance, beginning January 2018',
        'table_page':'https://www.transtats.bts.gov/TableInfo.asp?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGK',
        'field_reference':'https://transtats.bts.gov/Fields.asp?gnoyr_VQ=FGK',
        'use_terms':'Public BTS data; this manifest does not assert a separate reuse licence. Cite BTS and do not redistribute the raw archives from this project.',
        'selected_periods':{'fit':'2025-01 through 2025-04','validation_calibration':'2025-05','held_out_test':'2025-06'},
        'files':entries}
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Wrote data/manifest.json')

if __name__=='__main__':main()
