#!/usr/bin/env python3
"""Expert panel on the v6 task definition, BEFORE any SSL data is generated.

Four roles, each briefed to attack rather than approve, and each given the same spec. The prompt
says explicitly that approval is a valid verdict, so a unanimous rejection cannot be dismissed as
the prompt fishing for criticism -- but the instruction to look for failure modes is deliberate:
we want the objections now, not after generating a million rows.
"""
import json, os, re, sys, time, urllib.request

KEY = open(os.path.expanduser('~/.openai_key')).read().strip()
SPEC = open(os.path.expanduser('~/v6_task_spec.md')).read()
MODEL = 'gpt-5.4'

ROLES = {
 'ML-SSL': """You are a machine-learning scientist specialising in self-supervised representation
learning. You have published on masked modelling objectives and on when auxiliary pretraining
transfers to a downstream task and when it does not.""",
 'StructBio': """You are a computational structural biologist who builds structure-based drug design
methods. You know PDB data quality, pocket representation, interaction fingerprints, and the usual
ways geometric information is destroyed by careless featurisation.""",
 'MedChem': """You are a medicinal chemist who has run covalent hit-to-lead programmes. You care
whether a proposed task reflects how analogue design decisions are actually made at the bench.""",
 'LLM': """You are an NLP scientist who works on large language models for scientific data. You know
what transformers can and cannot compute in-context -- particularly arithmetic and geometric
reasoning over numeric tables -- and how tokenisation affects that.""",
}

QUESTION = """Below is a draft task definition for a covalent hit-to-lead model (7B LLM + LoRA).
Review it BEFORE any data is generated. Be specific and technical. Approval is a legitimate verdict
if the design is sound; do not invent objections. But if it has a fatal flaw, say so plainly.

Address, from your own expertise:
1. Is the central claim sound -- that masked-fragment restore on crystal complexes pretrains the
   same conditional as hit-to-lead design?
2. Will the SSL task actually FORCE use of the pocket, or can the model solve it from ligand context
   alone? What concretely would you do to guarantee the former?
3. Is the ATOMS block (raw interatomic distances + element + residue identity, no derived features)
   sufficient AND usable by a 7B LLM? If not, what minimum must be added?
4. What is the single biggest thing that will make this fail?

Answer as JSON:
{"verdict": "sound"|"sound with changes"|"flawed",
 "biggest_risk": "<one sentence>",
 "must_fix": ["<=4 specific items"],
 "answers": {"1": "...", "2": "...", "3": "...", "4": "..."}}

SPEC:
""" + SPEC


def call(sys_p):
    body = {'model': MODEL, 'messages': [{'role': 'system', 'content': sys_p},
                                         {'role': 'user', 'content': QUESTION}],
            'max_completion_tokens': 2200}
    req = urllib.request.Request('https://api.openai.com/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer %s' % KEY,
                                          'Content-Type': 'application/json'})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)['choices'][0]['message']['content']
        except Exception as e:
            if k == 4:
                return 'ERROR %s' % str(e)[:120]
            time.sleep(5 * (k + 1))


out = {}
for name, sysp in ROLES.items():
    txt = call(sysp)
    m = re.search(r'\{.*\}', txt or '', re.S)
    try:
        out[name] = json.loads(m.group(0))
    except Exception:
        out[name] = {'verdict': 'PARSE_FAIL', 'raw': (txt or '')[:400]}
    v = out[name]
    print('=' * 76)
    print('%s  ->  %s' % (name, v.get('verdict')))
    print('  biggest risk: %s' % v.get('biggest_risk', '-'))
    for f in (v.get('must_fix') or [])[:4]:
        print('   - %s' % f)
    print('  Q2 (does it force pocket use):')
    print('     %s' % (v.get('answers', {}).get('2', '-'))[:420])
    sys.stdout.flush()
json.dump(out, open(os.path.expanduser('~/panel_v6_out.json'), 'w'), indent=1)
print('\nverdicts: %s' % {k: v.get('verdict') for k, v in out.items()})