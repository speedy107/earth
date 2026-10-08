import {letterbox,decode} from './detector.mjs';
const $=id=>document.getElementById(id),video=$('video'),display=$('display'),ctx=display.getContext('2d');
const prep=document.createElement('canvas');prep.width=prep.height=320;const pc=prep.getContext('2d',{willReadFrequently:true});
const frame=document.createElement('canvas'),fc=frame.getContext('2d');
let session,stream,running=false,generation=0,busy=false,photoSource;
const setStatus=(text,error=false)=>{$('status').textContent=text;$('dot').style.background=error?'#ff666f':'#b5f2ce';};
function controls(){ $('start').disabled=!session||busy||running;$('stop').disabled=!running;$('photo').disabled=!session||busy||running; }
async function load(){
 try{
  if(!window.ort)throw new Error('Runtime download failed. Check your internet connection and reload.');
  ort.env.wasm.wasmPaths=new URL('./',location.href).href;ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.initTimeout=60000;
  setStatus('Downloading model (10 MB)…');
  const response=await fetch('./boxes-yolo11n.onnx');if(!response.ok)throw new Error(`Model download failed (${response.status}).`);
  setStatus('Preparing detector…');session=await ort.InferenceSession.create(await response.arrayBuffer(),{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  if(session.inputNames[0]!=='images'||!session.outputNames.includes('output0'))throw new Error('Model interface does not match this detector.');
  setStatus('Ready. Start your camera or choose a photo.');controls();
 }catch(e){setStatus(e.message,true);}
}
function stop(){running=false;generation++;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;controls();}
async function detect(source,width,height,token){
 const meta=letterbox(width,height);frame.width=width;frame.height=height;fc.drawImage(source,0,0,width,height);
 pc.fillStyle='rgb(114,114,114)';pc.fillRect(0,0,320,320);pc.drawImage(frame,meta.left,meta.top,meta.resizedW,meta.resizedH);
 const pixels=pc.getImageData(0,0,320,320).data,input=new Float32Array(3*320*320);
 for(let i=0;i<320*320;i++){input[i]=pixels[i*4]/255;input[320*320+i]=pixels[i*4+1]/255;input[2*320*320+i]=pixels[i*4+2]/255;}
 const tensor=new ort.Tensor('float32',input,[1,3,320,320]);let outputs;
 const started=performance.now();
 try{
  outputs=await session.run({images:tensor});
  if(token!==generation)return;
  const boxes=decode(outputs.output0.data,outputs.output0.dims,meta,Number($('threshold').value)/100);
  display.width=width;display.height=height;ctx.drawImage(frame,0,0);$('empty').hidden=true;
  const font=Math.max(15,Math.round(width/42));ctx.font=`bold ${font}px system-ui`;ctx.lineWidth=Math.max(2,width/180);
  for(const b of boxes){ctx.strokeStyle=b.cls?'#ff666f':'#45a7ff';ctx.fillStyle=ctx.strokeStyle;ctx.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);const label=`${b.cls?'Red':'Blue'} ${Math.round(b.score*100)}%`,tw=ctx.measureText(label).width+12,ty=Math.max(font+10,b.y1);ctx.fillRect(b.x1,ty-font-10,tw,font+10);ctx.fillStyle='#08121d';ctx.fillText(label,b.x1+6,ty-6);}
  const ms=performance.now()-started;$('speed').textContent=`${Math.round(ms)} ms / frame`;
  setStatus(boxes.length?`${boxes.length} block${boxes.length===1?'':'s'} detected`:'No blocks detected at this confidence.');
 }finally{tensor.dispose?.();if(outputs)Object.values(outputs).forEach(t=>t.dispose?.());}
}
async function loop(token){
 if(!running||token!==generation)return;
 busy=true;controls();
 try{if(video.readyState>=2)await detect(video,video.videoWidth,video.videoHeight,token);}
 catch(e){stop();setStatus(`Detection failed: ${e.message}`,true);}
 finally{busy=false;controls();}
 if(running&&token===generation)setTimeout(()=>loop(token),100);
}
$('start').onclick=async()=>{
 if(!session||busy||running)return;busy=true;controls();setStatus('Opening rear camera…');
 const token=++generation;
 try{
  if(!navigator.mediaDevices?.getUserMedia)throw new Error('Camera unavailable. Open the HTTPS site in Chrome or Safari.');
  stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'environment'},width:{ideal:640},height:{ideal:480}}});
  video.srcObject=stream;await video.play();photoSource=null;running=true;busy=false;controls();loop(token);
 }catch(e){stop();busy=false;controls();setStatus(e.name==='NotAllowedError'?'Camera permission denied. Allow camera access in browser settings, then try again.':e.message,true);}
};
$('stop').onclick=()=>{stop();setStatus('Camera stopped.');};
$('photo').onchange=async event=>{
 const file=event.target.files[0];if(!file||busy||running)return;busy=true;controls();const token=++generation;setStatus('Reading photo…');
 let url;
 try{url=URL.createObjectURL(file);const im=new Image();im.src=url;await im.decode();photoSource=im;await detect(im,im.naturalWidth,im.naturalHeight,token);}
 catch(e){setStatus(`Photo failed: ${e.message}`,true);}finally{if(url)URL.revokeObjectURL(url);busy=false;controls();event.target.value='';}
};
$('threshold').oninput=()=>{$('value').textContent=`${$('threshold').value}%`;};
$('threshold').onchange=async()=>{if(photoSource&&!busy&&!running){busy=true;controls();try{await detect(photoSource,photoSource.naturalWidth,photoSource.naturalHeight,generation);}catch(e){setStatus(e.message,true);}finally{busy=false;controls();}}};
document.addEventListener('visibilitychange',()=>{if(document.hidden&&running){stop();setStatus('Camera paused. Tap Start camera to resume.');}});
window.addEventListener('pagehide',stop);
window.addEventListener('load',load);
