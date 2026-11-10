"""Public-only SUMMER summaries, original task prompt and disk cache."""
from pathlib import Path
import sqlite3,json,os,time
from common import CFG,ROOT,objsha,write,sha

class Store:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.c = sqlite3.connect(path)
        self.c.execute('PRAGMA journal_mode=WAL')
        self.c.execute('PRAGMA cache_size=-8192')
        self.c.execute('CREATE TABLE IF NOT EXISTS items (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.c.commit()

    def get(self, key):
        row = self.c.execute('SELECT value FROM items WHERE key=?', (str(key),)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        self.c.execute('INSERT OR REPLACE INTO items VALUES (?,?)', (str(key), json.dumps(value)))

    def commit(self):
        self.c.commit()

    def close(self):
        self.c.commit(); self.c.close()


def summary(engine, names, public):
    """Summaries are public-only, shared across cells/seeds, bounded on disk."""
    store = Store(ROOT / 'prepared/public_summaries.sqlite')
    identity = objsha({'model': engine.identity, 'prompt': 'faithful-function-v1',
                       'public': {g: public.get(g, '') for g in names}})
    outputs = {}
    pending = []
    for gene in sorted(set(names)):
        text = public.get(gene)
        if not text:
            outputs[gene] = 'No concise public function available; this is not evidence of no response.'
            continue
        key = objsha([engine.identity, gene, text, 'summary-v1'])
        value = store.get(key)
        if value:
            outputs[gene] = value['text']
        else:
            pending.append((gene, key, text))
    for lo in range(0, len(pending), CFG['generation_batch']):
        part = pending[lo:lo+CFG['generation_batch']]
        prompts = [f'Human gene {g}. Public database description:\n{text}\n'
            'Summarize its molecular/cellular function in at most two short sentences. '
            'Use only the supplied facts. Do not invent downstream response directions, '
            'do not discuss the study, and do not add a preamble.' for g, key, text in part]
        for (gene, key, raw), value in zip(part, engine.generate(prompts, CFG['summary_tokens'])):
            # An incomplete summary must not silently become a fabricated fact.
            text = value['text'].strip() if value['ended'] and value['text'].strip() else raw
            outputs[gene] = text
            store.put(key, {'text': text, 'generated': value, 'source_fallback': text == raw})
        store.commit()
        write(ROOT / 'reports/summary_progress.json', {'status': 'RUNNING', 'completed_new': lo+len(part),
            'pending_new': len(pending), 'public_only': True, 'updated_unix': time.time()})
    store.close()
    write(ROOT / 'reports/summary_progress.json', {'status': 'PASS', 'used_genes': len(outputs),
        'public_only': True, 'scope_hash': identity})
    return outputs


def summer_prompt(data, row, summaries, retriever, part):
    t, q = int(row['target']), int(row['query'])
    facts = data.facts(t, q, summaries)
    examples = retriever.examples(t, q, part)
    def brief(gene):
        words = summaries.get(gene, 'Public function unavailable.').split()
        return ' '.join(words[:18]) + (' ...' if len(words) > 18 else '')
    observations = '\n'.join(f"CRISPRi {x['target']} -> query {x['query']}: {x['label']}. "
        f"Perturbed-gene function: {brief(x['target'])} Query-gene function: {brief(x['query'])}"
        for x in examples)
    return (facts + '\nStrict TRAIN experimental observations in the SAME cell line:\n' +
        (observations or 'No retrieved observation available.') + '\n'
        'Compare the perturbed genes and query genes in these observations with the requested genes. '
        'Consider overlapping functions and both supporting and conflicting response evidence. '
        'An observed response of a related pair is not a guaranteed response of this pair. '
        'Give a brief rationale (at most 45 words), then exactly one final tag '
        '<answer>up</answer>, <answer>same</answer>, or <answer>down</answer>. '
        'Choose the best supported population-average class; do not claim the examples prove causation.')


def code_identity():
    return {p.name: sha(p) for p in sorted((ROOT / 'code').glob('*')) if p.is_file()}

