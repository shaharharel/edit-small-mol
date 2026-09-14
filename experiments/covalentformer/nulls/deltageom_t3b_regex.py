"""Is the de-leak regex eating non-dummy bracket atoms?  r'\[\d+\*\]' requires digits THEN a literal *."""
import re,csv,collections
CF='/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer'
raw=list(csv.DictReader(open(CF+'/data/phaseB/phaseB.csv')))
pat=re.compile(r'\[\d+\*\]')
brk=re.compile(r'\[[^\]]*\]')
allb=collections.Counter(); eaten=collections.Counter(); ndiff=0
for r in raw:
    for col in ('keep','regrow','target','parent'):
        s=r.get(col) or ''
        for b in brk.findall(s):
            allb[b]+=1
            if pat.fullmatch(b): eaten[b]+=1
print('distinct bracket atoms in phaseB.csv keep/regrow/target/parent:')
for b,n in allb.most_common(30):
    print('   %-10s n=%-7d  %s'%(b,n,'STRIPPED by de-leak regex' if pat.fullmatch(b) else 'left intact'))
non_dummy_eaten=[b for b in eaten if '*' not in b]
print('\nNON-DUMMY bracket atoms the regex would strip: %d  -> %s'%(len(non_dummy_eaten),non_dummy_eaten))
print('OVER-DROP RISK: %s'%('NONE - regex only matches [<digits>*]' if not non_dummy_eaten else 'PRESENT'))
# collision check: does stripping merge two CHEMICALLY DISTINCT keep halves?
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
keeps={r['keep'] for r in raw if r.get('keep')}
g=collections.defaultdict(set)
for k in keeps: g[pat.sub('[*]',k)].add(k)
multi={s:v for s,v in g.items() if len(v)>1}
print('\nkeep halves: %d distinct raw -> %d distinct after stripping'%(len(keeps),len(g)))
print('  stripped forms covering >1 raw string: %d'%len(multi))
bad=0
for s,v in multi.items():
    cans={Chem.MolToSmiles(Chem.MolFromSmiles(pat.sub('[*]',x))) for x in v if Chem.MolFromSmiles(pat.sub('[*]',x))}
    if len(cans)>1: bad+=1
print('  of those, groups that are NOT a single canonical molecule (a genuine over-merge): %d'%bad)
