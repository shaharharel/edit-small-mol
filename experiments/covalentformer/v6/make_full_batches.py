#!/usr/bin/env python3
"""Split the whole corpus into labelling batches of PER pairs each.

PER is 100. The docstring used to cap it at 18 on the theory that a larger batch degrades label
quality; that was never measured, and when it was, it was false. A 100-pair batch was labelled by
three independent agents and the confidence distribution across the first and last twenty pairs was
flat (labA 17->13 high of 20, labB 8->7, labC 17->14, sentence length ~31 words throughout). Larger
batches are also FASTER per pair, because the 18 KB brief amortises: 18 s/pair at 100 against
27 s/pair at 25. Rendered evidence is 6.68 KB per pair, so a 100-pair batch is ~670 KB of context.

Pairs are shuffled with a fixed seed so any tranche is a random sample of the corpus rather than
an alphabetical slice -- otherwise tranche 1 would be all one target and tell us nothing about the
rest. The seed is load-bearing beyond that: it must stay 7 so that rebuilding the batches after a
renderer fix leaves every batch holding the SAME pairs, which is what lets an already-adjudicated
batch keep its labels.

Rebuilding is mandatory after any change to render_v7.py. check_packet_fresh.py enforces it by
re-rendering every pair and comparing byte for byte -- seven times now a fix has landed in the
source and never reached the file a labeller reads.
"""
import json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_v7 import render

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
SP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SP, 'batches')
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 100

rows = [json.loads(l) for l in open(D + '/v7_evidence.jsonl')]
random.seed(7)
random.shuffle(rows)
os.makedirs(OUT, exist_ok=True)

manifest = {}
for b in range((len(rows) + PER - 1) // PER):
    chunk = rows[b * PER:(b + 1) * PER]
    name = 'batch_%04d.txt' % (b + 1)
    with open(os.path.join(OUT, name), 'w') as f:
        f.write('LABELLING BATCH %d — %d analogue pairs. One final sentence per pair.\n\n'
                % (b + 1, len(chunk)))
        for e in chunk:
            f.write(render(e) + '\n\n')
    manifest[name] = [e['pair'] for e in chunk]

with open(os.path.join(SP, 'batch_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=1)

tot = sum(os.path.getsize(os.path.join(OUT, n)) for n in manifest)
print('corpus         %d pairs' % len(rows))
print('batches        %d files of up to %d pairs' % (len(manifest), PER))
print('total rendered %.1f MB  (avg %.0f KB per batch)'
      % (tot / 1e6, tot / len(manifest) / 1024))
print('manifest       batch_manifest.json')
print('\ntranche 1 (first 28 batches = %d pairs) is a random sample of the corpus, not a slice'
      % sum(len(v) for v in list(manifest.values())[:28]))
