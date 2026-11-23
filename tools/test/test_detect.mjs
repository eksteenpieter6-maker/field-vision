// Synthetic unit test for Field Vision sonar detection (simulation only, not hardware)
import {M,mkRaw,setBand} from './sim.mjs';
const sr=48000;
const SENS={STRICT:{N:6,cr:0,rel:0.006,rs:0.001,prev:2,oob:4},NORMAL:{N:4,cr:0,rel:0.003,rs:0.0005,prev:1,oob:8}};
const BAND={HIGH:[17500,20500],MID:[10000,14000],LOW:[4000,8000]};
let sd=7; const rnd=()=>{sd=(sd*16807)%2147483647; return sd/2147483647;};
function run(band,sens,scen,{dur=0.004,cfar}={}){
  setBand(...BAND[band],dur); const s=SENS[sens]; Object.assign(M.SON,{N:s.N,codeRatio:s.cr,relMin:s.rel,relMinSelf:s.rs,oobMax:s.oob}); if(cfar) M.SON.cfar=cfar;
  const tpls=[0,1,2].map(c=>M.makeChirp(c,sr)), secs=M.makeBand(sr), self={resp:[],n:[],learn:true};
  const hist=[], plotted=[], cnt={found:0,ripple:0,thr:0,gate:0,code:0,shape:0,ok:0}; let last=-1;
  for(let p=0;p<45;p++){ const code=last===0?1:0; last=code;
    const learnPhase=p<15, hdg=learnPhase? p*0.3 : 0.02*Math.sin(p);   // sweep while learning, then point at target
    self.learn=learnPhase;
    const opt={noise:0.0015*0.05,reverb:(scen==='echo'||scen==='room')?0.005:0};
    let ech=learnPhase?[{r:0.5+rnd()*3,g:0.4}]:[];
    if(scen==='echo'&&learnPhase) ech.push({r:1.0,g:0.8*(p%2)},{r:2.0,g:0.35*(p%2)});
    if(scen==='echo'&&!learnPhase) ech=[{r:1.0,g:0.8},{r:2.0,g:0.35}];
    if(scen==='clap'&&!learnPhase) opt.extra=(x,d0)=>{ const at=d0+Math.round((0.004+rnd()*0.02)*sr); for(let i=0;i<300;i++) x[at+i]+=(rnd()-0.5)*2*Math.exp(-i/60); };
    if(scen==='tone'&&!learnPhase) opt.extra=(x,d0)=>{ const f=(BAND[band][0]+BAND[band][1])/2+(rnd()-0.5)*1000, at=d0+Math.round(rnd()*0.01*sr); for(let i=0;i<Math.round(0.02*sr);i++) x[at+i]+=0.3*Math.sin(2*Math.PI*f*i/sr)*Math.sin(Math.PI*i/(0.02*sr)); };
    if(scen==='chirpnoise'&&!learnPhase) opt.extra=(x,d0)=>{ const t=M.makeChirp((code+1)%3,sr), at=d0+Math.round((0.004+rnd()*0.02)*sr); for(let i=0;i<t.length;i++) x[at+i]+=0.5*t[i]; }; // another device's sweep
    const r=M.detectPing(mkRaw(code,{echoes:ech,...opt}),code,tpls,secs,sr,null,self);
    if(!r.ok||learnPhase){ hist.push({hdg,peaks:r.peaks||[]}); if(hist.length>3) hist.shift(); continue; }
    for(const k in r.cnt) cnt[k]+=r.cnt[k];
    const ok=M.confirmPeaks(r.peaks,hdg,hist,s.prev); r.peaks.forEach((pk,i)=>{ if(ok[i]) plotted.push(pk.r); });
    hist.push({hdg,peaks:r.peaks}); if(hist.length>3) hist.shift();
  }
  return {plotted,cnt};
}
let fail=0;
const near=(arr,x)=>arr.filter(r=>Math.abs(r-x)<0.06).length;
for(const band of ['MID','HIGH','LOW']) for(const sens of ['STRICT','NORMAL']){
  const room=run(band,sens,'room').plotted, e=run(band,sens,'echo'), c=run(band,sens,'clap'), t=run(band,sens,'tone'), o=run(band,sens,'chirpnoise');
  const n1=near(e.plotted,1.0), n2=near(e.plotted,2.0), stray=e.plotted.filter(r=>Math.abs(r-1)>=0.06&&Math.abs(r-2)>=0.06&&!room.some(q=>Math.abs(q-r)<0.06)).length;
  const pass = n1>=15 && n2>=10 && stray<=3 && c.plotted.length<=1 && t.plotted.length<=1 && (sens!=='STRICT' || o.plotted.length<=1);
  if(!pass) fail++;
  console.log(`room-only reflections plotted at: ${[...new Set(room.map(r=>r.toFixed(1)))].join(',')} m`);
  console.log(`${pass?'PASS':'FAIL'} ${band.padEnd(4)} ${sens.padEnd(6)} echo@1.0m ${n1}/30 echo@2.0m ${n2}/30 stray ${stray} (mean ${ (e.plotted.filter(r=>Math.abs(r-1)<0.06).reduce((a,b)=>a+b,0)/Math.max(1,n1)).toFixed(3)} m, ${(e.plotted.filter(r=>Math.abs(r-2)<0.06).reduce((a,b)=>a+b,0)/Math.max(1,n2)).toFixed(3)} m) | clap plotted ${c.plotted.length} tone ${t.plotted.length} foreign-chirp ${o.plotted.length} | cnt ${JSON.stringify(e.cnt)}`);
}
process.exit(fail?1:0);
