// Simulated phone sonar: loads DSP block from index.html, runs realistic room sim
import fs from 'fs';
const file=process.env.FV||new URL('../../index.html',import.meta.url).pathname;
const src=fs.readFileSync(file,'utf8'); const blk=src.split('// ==SONAR-DSP-BEGIN==')[1].split('// ==SONAR-DSP-END==')[0];
const M=new Function(blk+'\nreturn {SON,makeChirp,makeBand,detectPing,corrAt,confirmPeaks};')();
export {M};
const sr=48000;
let seed=1; const rnd=()=>{seed=(seed*16807)%2147483647; return seed/2147483647;}; const gn=()=>{let u=rnd()||1e-9,v=rnd(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
let rs=99; const r2=()=>{rs=(rs*16807)%2147483647; return rs/2147483647;};
const ROOM=[]; for(let k=0;k<300;k++){ const r=0.3+r2()*8; let u=r2()||1e-9,v=r2(); ROOM.push([r, Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)*Math.exp(-r/2.5)/Math.max(0.5,r)]); }
export function setBand(f0,f1,dur){ M.SON.f0=f0; M.SON.f1=f1; M.SON.dur=dur; }
// room impulse: direct (gain 1, with speaker ringdown), discrete echoes [{r,g}], diffuse reverb
export function mkRaw(code,{echoes=[],reverb=0.02,noise=3e-5,lat=2000,ring=true,extra=null,dirJit=0.02}={}){
  const pre=Math.round(M.SON.pre*sr), n=Math.round((M.SON.pre+M.SON.post)*sr), x=new Float64Array(n);
  const t=M.makeChirp(code,sr);
  // speaker resonance ringdown kernel
  const ir=new Float64Array(Math.round(0.004*sr)); ir[0]=1; if(ring) for(let i=1;i<ir.length;i++) ir[i]=0.3*Math.exp(-i/(0.0008*sr))*Math.sin(2*Math.PI*11500*i/sr);
  const st=new Float64Array(t.length+ir.length); for(let i=0;i<t.length;i++) for(let j=0;j<ir.length;j++) st[i+j]+=t[i]*ir[j];
  const add=(d,g)=>{ const i0=Math.floor(d), fr=d-i0; for(let i=0;i<st.length;i++){ const k=pre+lat+i0+i; if(k+1<n){ x[k]+=g*st[i]*(1-fr); x[k+1]+=g*st[i]*fr; } } };
  add(0,50*(1+dirJit*gn()));                                  // direct
  add(0.00015*sr,10*(1+dirJit*gn()));                          // body path
  for(const e of echoes) add(e.r*2/343*sr, e.g);
  if(reverb){ // diffuse room: fixed set of small reflections (static room, same every ping) + slight per-ping jitter
    for(const [r,g0] of ROOM) add(r*2/343*sr, reverb*50*g0*(1+0.1*gn())); }
  for(let i=0;i<n;i++) x[i]+=noise*gn();
  if(extra) extra(x,pre+lat);
  return Float32Array.from(x);
}
