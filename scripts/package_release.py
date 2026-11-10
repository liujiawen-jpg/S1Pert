"""Build a deterministic anonymous ZIP: no owner IDs, history or local timestamps."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from check_release import ROOT,audit,source_files

def entry(name):
    z=zipfile.ZipInfo('s1pert/'+name,date_time=(1980,1,1,0,0,0))
    z.create_system=3;z.external_attr=0o100644<<16;z.compress_type=zipfile.ZIP_DEFLATED
    z.extra=b'';z.comment=b'';return z

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    out=p.parse_args().out.resolve()
    if out.is_relative_to(ROOT) or out.exists():p.error('Choose a new archive outside the source tree')
    result=audit()
    if result['status']!='PASS':raise SystemExit(json.dumps(result))
    files=list(source_files());hashes={}
    for path in files:
        content=path.read_bytes();hashes[str(path.relative_to(ROOT))]={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    manifest={'schema':'s1pert.source.v1','weights_included':False,'raw_data_included':False,
              'timestamps':'normalized','git_history_included':False,'files':hashes}
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x') as z:
        for path in files:z.writestr(entry(str(path.relative_to(ROOT))),path.read_bytes())
        z.writestr(entry('RELEASE_MANIFEST.json'),json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
        for name,info in hashes.items():assert hashlib.sha256(z.read('s1pert/'+name)).hexdigest()==info['sha256']
    digest=hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(out.suffix+'.sha256').write_text(digest+'  '+out.name+'\n')
    print(json.dumps({'archive':out.name,'bytes':out.stat().st_size,'files':len(files)+1,'sha256':digest,'status':'PASS'},indent=2))

if __name__=='__main__':main()
