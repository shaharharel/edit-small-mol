"""TARGET 1b: per-channel permutation on the Z-SCORED vs RAW Phase B model."""
import sys,os,json,copy,random,statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # self-contained: replay.py sits beside this file
import replay, torch
CF="/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer"
SEEDS=[404,505,606,707,808,909]
CH={'d':'d','cos_theta':'cos_theta','v_free':'v_free'}
NDRAW=20
# FIXED CHANNEL INDEX, replacing hash(cname). Python string hash() is PER-PROCESS RANDOMISED when
# PYTHONHASHSEED is unset -- measured hash('d')%1000 = 570, 411, 218 in three consecutive
# interpreters -- so the seed, and therefore the ENTIRE 48-cell table behind #145, could not be
# regenerated. That is a violation of this project's own seeding rule inside the harness built to
# enforce it, and it is why the stored draws in perm_results.json were the only record. The stored
# table remains valid: every aggregate was independently revalidated from its raw 20-draw arrays.
_CHIDX = {'d': 0, 'cos_theta': 1, 'v_free': 2, 'ALL3': 3}
out={}
for arm_tag,arm_name in [('Bz','zscored'),('Bconv','raw')]:
    for s in SEEDS:
        d=os.path.join(CF,f"ckpt_{arm_tag}_from_a_s{s}")
        eps=sorted([f for f in os.listdir(d) if f.startswith('ep') and f.endswith('.ckpt')],key=lambda f:int(f[2:-5]))
        # pick the ARGMIN epoch (the selected model), read from the last ckpt's history
        ckl=torch.load(os.path.join(d,eps[-1]),map_location='cpu',weights_only=False)
        vals=[e['val'] for e in ckl['history']]
        best=vals.index(min(vals))
        # PASS THE CHECKPOINT'S OWN FILTER VERSION. Absent -> 1, correct for every pre-14:02 arm.
        # This is what stops the assert below from being a tripwire: it now compares a replay built
        # the SAME way as the checkpoint, rather than the newest way.
        r=replay.replay(s, deleak_version=int(ckl.get('deleak_version', 1)))
        gstats=r['gstats'] if arm_tag=='Bz' else None
        m,ck,vocab,tok=replay.load_model(os.path.join(d,f'ep{best}.ckpt'))
        L0=replay.eval_loss(m,r['va'],vocab,tok,gstats)
        assert abs(L0-vals[best])<1e-9, (L0,vals[best])
        va=r['va']; n=len(va)
        rec={'epoch':best,'L0':L0,'n_va':n,'n_tr':len(r['tr']),'gzscore':ck.get('gzscore')}
        for cname in list(CH)+['ALL3']:
            deltas=[]
            for p in range(NDRAW):
                rng=random.Random(1000*s+97*p+7*_CHIDX[cname])
                rows2=[dict(x) for x in va]
                targets=[cname] if cname!='ALL3' else list(CH)
                for t in targets:
                    vs=[x[t] for x in va]
                    idx=list(range(n)); rng.shuffle(idx)
                    for i,j in enumerate(idx): rows2[i][t]=vs[j]
                Lp=replay.eval_loss(m,rows2,vocab,tok,gstats)
                deltas.append(Lp-L0)
            rec[cname]={'mean':st.mean(deltas),'sd':st.stdev(deltas),
                        'se':st.stdev(deltas)/len(deltas)**0.5,
                        'min':min(deltas),'max':max(deltas),
                        'frac_pos':sum(1 for x in deltas if x>0)/len(deltas),
                        'draws':deltas}
            print(f"{arm_name} s{s} ep{best} n={n} {cname:10s}: mean {rec[cname]['mean']:+.5f} "
                  f"se {rec[cname]['se']:.5f} range [{min(deltas):+.5f},{max(deltas):+.5f}] "
                  f"pos {rec[cname]['frac_pos']:.2f}",flush=True)
        out[f"{arm_name}_s{s}"]=rec
        # WRITE BESIDE THIS SCRIPT, NOT INTO /tmp. Moving the script into the repo while leaving its
# OUTPUT in /tmp is a half-fix: the harness survives a reboot and the numbers do not. That is
# the fourth variant of the same defect tonight (0.2917, the length-matched null, the
# delta-geometry 88,891, and this).
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'perm_results.json')
json.dump(out, open(_OUT, 'w'), indent=1)
print('  wrote %s' % _OUT)
print("DONE")
