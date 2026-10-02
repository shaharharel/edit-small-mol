CSS = r"""<style>
:root{
 --paper:#fbfbf9; --ink:#16181a; --mu:#5a6066; --faint:#8b9197;
 --rule:#c9ccc6; --hair:#e3e5e0; --code:#f2f3ef;
 --ok:#1f6b45; --no:#a33127; --link:#1f4e79;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --paper:#131517; --ink:#e4e6e2; --mu:#9aa1a7; --faint:#757c82;
 --rule:#3a3f44; --hair:#24282c; --code:#1a1e21;
 --ok:#63b98c; --no:#dd7b6e; --link:#7fb0dc;}}
:root[data-theme="dark"]{
 --paper:#131517; --ink:#e4e6e2; --mu:#9aa1a7; --faint:#757c82;
 --rule:#3a3f44; --hair:#24282c; --code:#1a1e21;
 --ok:#63b98c; --no:#dd7b6e; --link:#7fb0dc;}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);
 font:16px/1.62 "Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif}
.w{max-width:860px;margin:0 auto;padding:56px 24px 96px}
.hd{border-bottom:2px solid var(--ink);padding-bottom:16px;margin-bottom:26px}
h1{font-size:30px;line-height:1.16;margin:0 0 8px;font-weight:600;letter-spacing:-.01em;text-wrap:balance}
.byline{font:12.5px/1.5 ui-sans-serif,-apple-system,sans-serif;color:var(--mu);letter-spacing:.02em}
h2{font:600 18px/1.3 ui-sans-serif,-apple-system,sans-serif;margin:42px 0 10px;letter-spacing:-.005em}
h3{font:600 14.5px/1.35 ui-sans-serif,-apple-system,sans-serif;margin:26px 0 7px;color:var(--ink)}
h2 .sn,h3 .sn{color:var(--faint);font-weight:500;margin-right:.5em;font-variant-numeric:tabular-nums}
p{margin:0 0 11px}
.abs{border:1px solid var(--rule);background:var(--code);padding:17px 20px;margin:0 0 8px;font-size:15px}
.abs h4{font:600 11px/1 ui-sans-serif,sans-serif;letter-spacing:.14em;text-transform:uppercase;
 color:var(--mu);margin:0 0 9px}
.abs p:last-child{margin-bottom:0}
code,pre,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
code{font-size:.88em;background:var(--code);padding:1px 4px;border-radius:2px}
a{color:var(--link)}
/* --- tables: booktabs --- */
.tbl{margin:16px 0 22px}
.cap{font:12.5px/1.5 ui-sans-serif,-apple-system,sans-serif;color:var(--mu);margin:0 0 7px}
.cap b{color:var(--ink);font-weight:600}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font:13.5px/1.45 ui-sans-serif,-apple-system,sans-serif}
table thead tr:first-child th{border-top:1.5px solid var(--ink)}
thead th{border-bottom:1px solid var(--ink);padding:7px 10px;text-align:left;font-weight:600;
 vertical-align:bottom}
tbody td{padding:6px 10px;border-bottom:1px solid var(--hair)}
tbody tr:last-child td{border-bottom:1.5px solid var(--ink)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace}
tbody tr.rule-above td{border-top:1px solid var(--rule)}
.em{font-weight:650}
.ok{color:var(--ok);font-weight:650} .no{color:var(--no);font-weight:650}
/* --- figures --- */
figure{margin:18px 0 22px}
figure>figcaption{font:12.5px/1.5 ui-sans-serif,-apple-system,sans-serif;color:var(--mu);margin-top:9px}
figure>figcaption b{color:var(--ink);font-weight:600}
.box{border:1px solid var(--rule);padding:16px;background:var(--code)}
/* rung diagram */
.rungs{display:grid;grid-template-columns:repeat(3,1fr);gap:0;border:1px solid var(--rule)}
.rung{padding:13px 14px;border-right:1px solid var(--rule);background:var(--paper)}
.rung:last-child{border-right:0}
.rung .rn{font:600 10.5px/1.3 ui-sans-serif,sans-serif;letter-spacing:.1em;text-transform:uppercase;color:var(--mu)}
.rung .rt{font:600 14.5px/1.3 ui-sans-serif,sans-serif;margin:5px 0 6px}
.rung .rd{font-size:13.5px;color:var(--mu);line-height:1.5}
.rung .rc{margin-top:9px;font-family:ui-monospace,monospace;font-size:11.5px;color:var(--ink);
 border-top:1px solid var(--hair);padding-top:7px}
.combine{display:grid;grid-template-columns:1fr 1fr;gap:0;border:1px solid var(--rule);border-top:0}
.cmb{padding:12px 14px;border-right:1px solid var(--rule);font-size:13.5px}
.cmb:last-child{border-right:0}
.cmb b{font:600 13.5px/1.3 ui-sans-serif,sans-serif;display:block;margin-bottom:3px}
/* prompt/response exchange blocks */
.xch{border:1px solid var(--rule);margin:9px 0;background:var(--paper)}
.xch>.xh{display:flex;justify-content:space-between;align-items:center;gap:8px;
 padding:6px 11px;border-bottom:1px solid var(--hair);background:var(--code);
 font:600 11px/1.4 ui-sans-serif,sans-serif;letter-spacing:.06em;text-transform:uppercase;color:var(--mu)}
.xch .verdict{letter-spacing:.04em}
.xr{display:grid;grid-template-columns:78px 1fr;border-bottom:1px solid var(--hair)}
.xr:last-child{border-bottom:0}
.xr>.k{padding:7px 10px;font:600 10.5px/1.5 ui-sans-serif,sans-serif;letter-spacing:.07em;
 text-transform:uppercase;color:var(--faint);border-right:1px solid var(--hair);background:var(--code)}
.xr>.v{padding:7px 11px;font-family:ui-monospace,monospace;font-size:12px;line-height:1.55;
 white-space:pre-wrap;word-break:break-word;overflow-x:auto}
.v.g-ok{color:var(--ok)} .v.g-no{color:var(--no)}
/* case panels */
.tri{display:grid;grid-template-columns:repeat(3,1fr);gap:0;border:1px solid var(--rule)}
.tri>figure{margin:0;padding:10px;border-right:1px solid var(--rule);text-align:center}
.tri>figure:last-child{border-right:0}
.tri figcaption{font:600 10px/1.3 ui-sans-serif,sans-serif;letter-spacing:.08em;text-transform:uppercase;
 color:var(--faint);margin:0 0 7px}
.tri code{display:block;margin-top:6px;font-size:10px;color:var(--mu);word-break:break-all;text-align:left}
.mol{background:#fff;padding:4px;border:1px solid var(--hair)}
.mol svg{max-width:100%;height:auto;display:block;margin:0 auto}
.case{margin:0 0 20px}
.ch{display:flex;gap:9px;align-items:baseline;flex-wrap:wrap;
 font:12.5px/1.5 ui-sans-serif,sans-serif;margin-bottom:7px}
.ch .idx{font-weight:700;font-family:ui-monospace,monospace}
.ch .meta{color:var(--mu)}
details{margin-top:8px}
summary{cursor:pointer;font:12.5px/1.5 ui-sans-serif,sans-serif;color:var(--link)}
summary:focus-visible{outline:2px solid var(--link);outline-offset:2px}
details pre{margin:7px 0 0;background:var(--code);border:1px solid var(--hair);padding:9px;
 font-size:11.5px;white-space:pre-wrap;word-break:break-word;max-height:200px;overflow:auto}
.two{display:grid;grid-template-columns:1fr 1fr;gap:9px}
.lab{font:600 10px/1.4 ui-sans-serif,sans-serif;letter-spacing:.08em;text-transform:uppercase;
 color:var(--faint);margin-bottom:3px}
ul,ol{margin:0 0 11px;padding-left:22px} li{margin-bottom:5px}
.note{border-left:2px solid var(--rule);padding:2px 0 2px 15px;margin:14px 0;color:var(--mu);font-size:15px}
.note b{color:var(--ink)}
.ft{margin-top:56px;border-top:1px solid var(--rule);padding-top:14px;
 font:12px/1.6 ui-sans-serif,sans-serif;color:var(--faint)}
@media(max-width:760px){.rungs,.combine,.tri,.two{grid-template-columns:1fr}
 .rung,.cmb{border-right:0;border-bottom:1px solid var(--rule)}
 .tri>figure{border-right:0;border-bottom:1px solid var(--rule)}
 .xr{grid-template-columns:1fr}.xr>.k{border-right:0;border-bottom:1px solid var(--hair)}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
@media print{body{background:#fff}.w{max-width:none;padding:0}}

.verdict{font-size:13px}
.pill{font:600 9.5px/1.4 ui-monospace,monospace;letter-spacing:.07em;text-transform:uppercase;
 padding:2px 7px;border-radius:2px;background:var(--hair);color:var(--mu)}
.pill--hit{background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok)}
.flow{border:1px solid var(--rule);margin:14px 0 6px}
.flow>.fh{background:var(--code);border-bottom:1px solid var(--rule);padding:8px 12px;
 font:600 12.5px/1.4 ui-sans-serif,sans-serif}
.fh .tag{font:600 9.5px/1.4 ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase;
 color:var(--mu);margin-left:8px}
.fstep{display:grid;grid-template-columns:118px 1fr;border-bottom:1px solid var(--hair)}
.fstep:last-child{border-bottom:0}
.fstep>.sn2{padding:9px 10px;border-right:1px solid var(--hair);background:var(--code);
 font:600 10px/1.45 ui-sans-serif,sans-serif;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
.fstep>.sv{padding:9px 12px;font-family:ui-monospace,monospace;font-size:11.5px;line-height:1.55;
 white-space:pre-wrap;word-break:break-word;overflow-x:auto}
.sv .an{font-family:ui-sans-serif,sans-serif;color:var(--mu);font-size:12px}
.sv.inp{background:color-mix(in srgb,var(--rule) 22%,transparent)}
.sv.outp{background:color-mix(in srgb,var(--ok) 9%,transparent)}
</style>"""
