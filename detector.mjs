export function letterbox(width,height,size=320){
  const scale=Math.min(size/width,size/height);
  const resizedW=Math.round(width*scale),resizedH=Math.round(height*scale);
  return {width,height,scale,resizedW,resizedH,left:Math.round((size-resizedW)/2-0.1),top:Math.round((size-resizedH)/2-0.1)};
}
export function iou(a,b){
  const intersect=Math.max(0,Math.min(a.x2,b.x2)-Math.max(a.x1,b.x1))*Math.max(0,Math.min(a.y2,b.y2)-Math.max(a.y1,b.y1));
  const union=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-intersect;
  return union>0?intersect/union:0;
}
export function decode(data,dims,meta,threshold=.25,nms=.45){
  if(dims.length!==3||dims[0]!==1||dims[1]!==6||dims[2]!==2100)throw new Error(`Unexpected model output: ${dims}`);
  const n=dims[2], candidates=[];
  for(let i=0;i<n;i++){
    const blue=data[4*n+i],red=data[5*n+i],cls=red>blue?1:0,score=Math.max(blue,red);
    if(!Number.isFinite(score)||score<threshold)continue;
    const cx=data[i],cy=data[n+i],w=data[2*n+i],h=data[3*n+i];
    if(![cx,cy,w,h].every(Number.isFinite)||w<=0||h<=0)continue;
    const clamp=(v,max)=>Math.max(0,Math.min(max,v));
    const box={cls,score,x1:clamp((cx-w/2-meta.left)/meta.scale,meta.width),y1:clamp((cy-h/2-meta.top)/meta.scale,meta.height),x2:clamp((cx+w/2-meta.left)/meta.scale,meta.width),y2:clamp((cy+h/2-meta.top)/meta.scale,meta.height)};
    if(box.x2>box.x1&&box.y2>box.y1)candidates.push(box);
  }
  candidates.sort((a,b)=>b.score-a.score);
  const keep=[];
  for(const box of candidates){if(!keep.some(k=>k.cls===box.cls&&iou(k,box)>nms))keep.push(box);if(keep.length>=100)break;}
  return keep;
}
