"""Fail-closed anonymous source audit. Never prints matching secret values."""
import ast
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
IGNORED={'.git','.venv','__pycache__','.pytest_cache','build','dist','assets','workspace','external'}
ALLOWED={'.py','.md','.txt','.json','.toml','.yaml','.yml','.csv','.tsv','.sh','.example'}
EXCLUDE={'RELEASE_MANIFEST.json'}
PATTERNS={
 'local_absolute_path':r'/(?:root|mnt|tmp|home|Users)/[A-Za-z0-9_.-]+',
 'windows_user_path':r'[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]',
 'email_address':r'\b[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
 'ip_address':r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
 'gpu_uuid':r'GPU-[a-f0-9]{8}-[a-f0-9-]{20,}',
 'credential':r'(?:hf_[A-Za-z0-9]{25,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{30,})',
 'private_key':r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
 'url_credentials':r'https?://[^\s/]+:[^\s/]+@',
}

def source_files():
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if any(x in IGNORED or x.endswith('.egg-info') for x in rel.parts):continue
        if path.name in EXCLUDE:continue
        if path.is_symlink() or path.is_file():yield path

def audit():
    files=list(source_files());issues=[]
    for path in files:
        rel=str(path.relative_to(ROOT))
        if path.is_symlink():issues.append([rel,'symlink']);continue
        if path.stat().st_size>5*2**20:issues.append([rel,'size_limit']);continue
        if path.suffix not in ALLOWED and path.name not in {'.gitignore','.gitattributes','LICENSE','NOTICE'}:
            issues.append([rel,'unapproved_file_type']);continue
        try:content=path.read_text(encoding='utf-8')
        except UnicodeError:issues.append([rel,'binary_payload']);continue
        if path.suffix=='.py':
            try:ast.parse(content,filename=rel)
            except SyntaxError:issues.append([rel,'python_syntax'])
        if path.suffix=='.json':
            try:json.loads(content)
            except ValueError:issues.append([rel,'json_syntax'])
        for name,pattern in PATTERNS.items():
            if re.search(pattern,content,flags=re.I):issues.append([rel,name])
    return {'status':'PASS' if not issues else 'FAIL','source_files':len(files),
            'scope':'UTF-8 source, paths, credentials, contacts and archive payload policy',
            'weights_or_data_required':False,'issues':issues}

def main():
    result=audit();print(json.dumps(result,indent=2));return int(result['status']!='PASS')

if __name__=='__main__':raise SystemExit(main())
