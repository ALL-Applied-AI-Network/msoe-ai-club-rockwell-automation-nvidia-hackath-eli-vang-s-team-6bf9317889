"""Portable HTML review page and explanatory image overlays."""
import base64
import io
import json
import hashlib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

COLORS={'pass':'#20a875','missing_cap':'#f29a32','crooked_cap':'#bf78ec',
        'damaged_vial':'#f15b65','no_vial':'#60a5fa',None:'#aab4c4'}
SHORT={'pass':'PASS','missing_cap':'CAP','crooked_cap':'TILT','damaged_vial':'DMG',
       'no_vial':'EMPTY',None:'REVIEW'}


def font(size=20):
    for name in ['/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(name).exists(): return ImageFont.truetype(name,size)
    return ImageFont.load_default()


def overlay(im,result):
    out=Image.new('RGB',(1920,1220),'#101c2d');out.paste(im,(0,100))
    d=ImageDraw.Draw(out)
    reviewed=sum(s['review_status']=='confirmed' for s in result['slots'])
    d.text((30,20),f'{result["filename"]}   |   {reviewed}/60 POSITIONS CONFIRMED',font=font(29),fill='white')
    d.text((30,60),'Five rows per column. G = good, S = scrap. Numbering starts at the top right.',font=font(21),fill='#becde1')
    for s in result['slots']:
        x,y=s['center'];y+=100;c=COLORS[s['class']]
        d.ellipse((x-24,y-24,x+24,y+24),outline=c,width=4)
        label=s['id']
        d.rectangle((x-24,y+23,x+27,y+46),fill='#101c2d')
        d.text((x-21,y+25),label,font=font(16),fill=c)
    x=30
    for k,c in COLORS.items():
        d.rectangle((x,1187,x+17,1204),fill=c)
        d.text((x+26,1183),(k or 'needs review').replace('_',' '),font=font(20),fill='white')
        x+=290
    return out


def contact_sheet(im,result,area):
    width,height=1200,1060
    out=Image.new('RGB',(width,height),'#101c2d');d=ImageDraw.Draw(out)
    d.text((24,18),f'{result["filename"]}  |  {area.upper()} POSITIONS',font=font(25),fill='white')
    d.text((24,56),'Read each column downward, starting at the right. Crops include neighboring context.',font=font(18),fill='#bdc9dc')
    for s in [s for s in result['slots'] if s['area']==area]:
        col=6-s['column_from_right'];row=s['row']-1
        x=col*200+8;y=row*190+100
        crop=marked_crop(im,s);crop.thumbnail((182,137))
        out.paste(crop,(x+(184-crop.width)//2,y))
        c=COLORS[s['class']]
        d.rectangle((x,y,x+184,y+178),outline=c,width=2)
        d.text((x+7,y+145),f'{s["id"]}  {SHORT[s["class"]]}',font=font(19),fill=c)
    return out


def marked_crop(im,slot):
    crop=im.crop(slot['bbox'])
    d=ImageDraw.Draw(crop)
    x=slot['center'][0]-slot['bbox'][0];y=slot['center'][1]-slot['bbox'][1]
    d.ellipse((x-21,y-21,x+21,y+21),outline=COLORS[slot['class']],width=1)
    return crop


def data_uri(im, max_size=None):
    im=im.copy()
    if max_size:im.thumbnail(max_size)
    buf=io.BytesIO();im.save(buf,format='JPEG',quality=88)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()


def build_report(results,path):
    from labeler import VERSION
    assets={}
    for r in results:
        im=ImageOps.exif_transpose(Image.open(Path(path).parent/r['source_path'])).convert('RGB')
        assets[r['filename']]=dict(scene=data_uri(im,(1500,844)),
            crops={s['id']:data_uri(marked_crop(im,s)) for s in r['slots']})
    hashes={r['model_hash'] for r in results}
    if len(hashes)!=1: raise ValueError('A review page must contain one model version')
    # Preserve the stored model identity when refreshing a page's presentation,
    # so its existing browser-local review edits are still found.
    revision=hashlib.sha256(json.dumps(results,sort_keys=True).encode()).hexdigest()
    payload=json.dumps(dict(schema_version=VERSION,model_hash=results[0]['model_hash'],
        review_revision=revision,review_timestamp=results[0].get('review_timestamp'),results=results)).replace('<', chr(92)+'u003c')
    replacements = {'__PAYLOAD__':payload, '__ASSETS__':json.dumps(assets).replace('<', chr(92)+'u003c'), '__COLORS__':json.dumps(COLORS)}
    import re
    html = re.sub(r'__PAYLOAD__|__ASSETS__|__COLORS__', lambda match: replacements[match.group()], HTML)
    Path(path).write_text(html)


HTML=r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vial inspection review</title>
<style>
:root{color-scheme:dark;font-family:Arial,sans-serif;background:#0b1320;color:#e5edf8}*{box-sizing:border-box}body{margin:0}button,select,input,textarea{font:inherit}button,select{background:#23344b;color:#fff;border:1px solid #52627a;border-radius:6px;padding:9px;cursor:pointer}button:hover{background:#344d6e}button:focus-visible,select:focus-visible{outline:3px solid #7edbd0}header{padding:24px 30px;background:#132236;display:flex;gap:20px;align-items:center;justify-content:space-between}h1{font-size:24px;margin:0 0 7px}p{line-height:1.5}.sub{font-size:14px;color:#b9c8dc}.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:16px 30px}.badge{border:1px solid #736944;background:#352e1a;padding:8px;border-radius:5px;color:#ffe0a0;font-size:13px}.layout{display:grid;grid-template-columns:minmax(560px,1fr) 340px;gap:22px;padding:0 30px 30px}.scene{width:100%;background:#132236;border-radius:8px}.groups{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}.card{background:#142337;border:1px solid #30445e;border-radius:8px;padding:15px}.card h2{margin:0 0 10px;font-size:18px}.flag{display:inline-block;margin:3px 4px 3px 0;padding:5px 7px;border-radius:4px;background:#2f4057;font-size:13px}.yes{background:#5b3039}.unknown{background:#473e26}.fine{color:#bdc9dc;font-size:13px;line-height:1.5}.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:7px}.tile{padding:5px;min-width:0;background:#142337;border:2px solid #53657a;border-radius:6px}.tile img{width:100%;height:115px;object-fit:contain}.tile span{font-size:12px;display:block;white-space:nowrap}.tile.active{outline:3px solid #fff}.detail{position:sticky;top:12px;align-self:start}.detail img{width:100%;height:290px;object-fit:contain;background:#080e17;border-radius:6px}.detail label{display:block;margin:16px 0 6px}.detail select,.detail textarea{width:100%}textarea{background:#0e1b2c;color:#fff;border:1px solid #52627a;padding:8px;height:96px}.legend{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;font-size:12px}.legend i{display:inline-block;width:10px;height:10px;margin-right:4px}#notice{padding:10px;background:#383522;border-radius:4px;display:none}.svgslot{cursor:pointer}.svgslot:hover{stroke-width:7}a{color:#84ccff}@media(max-width:950px){.layout{grid-template-columns:1fr;padding:0 12px}.detail{position:static}.toolbar,header{padding:16px}.grid{grid-template-columns:repeat(3,1fr)}.groups{grid-template-columns:1fr}}@media print{header,.toolbar,.detail{display:none}.layout{display:block}.tile{break-inside:avoid}}
.position-nav{display:flex;gap:10px;margin-top:10px}.position-nav button{flex:1}.position-nav button:disabled{opacity:.4;cursor:default}#positionProgress{margin:7px 0 0}
</style>
<header><div><h1>Vial inspection review</h1><div class="sub">Camera appearance · slot occupancy · tray fill sequence</div><div id="snapshotTime" class="sub"></div></div><span class="badge">CHECK POSITION REVIEW STATUS</span></header>
<div class="toolbar"><label for="imagePick">Image</label><select id="imagePick"></select><select id="areaPick" aria-label="Tray"><option value="good">Good tray</option><option value="scrap">Scrap tray</option></select><button id="next">Next unconfirmed</button><button id="loadReview">Load review JSON</button><input id="reviewFile" type="file" accept=".json,application/json" hidden><button id="download">Export review JSON</button><span id="progress" class="sub"></span><span id="loadStatus" class="sub" role="status" aria-live="polite"></span></div>
<div class="layout"><main><svg id="scene" class="scene" viewBox="650 300 1030 780" role="img" aria-label="Inspection photo with numbered tray positions"></svg><div class="legend" id="legend"></div><div class="legend" aria-label="Review status legend"><span><b style="background:#fbbf24;color:#101c2d;border-radius:50%;padding:2px 7px">!</b> Awaiting review · dashed ring</span><span><b style="background:#16a34a;color:white;border-radius:50%;padding:2px 5px">✓</b> Reviewed · solid ring</span></div><p class="fine" id="selectionReason"></p><div class="groups" id="groups"></div><p class="fine">Tray findings below reflect the saved run. After edits, import the exported review JSON to recompute them. A visually passing scrap vial does not prove an incorrect upstream rejection.</p><div id="notice">Labels have changed. Export and import your review to recompute tray findings.</div><h2 style="font-size:19px">Positions in physical tray order</h2><p class="fine">Start at the top right and move down five rows before the next column to the left. Each tile is a context crop; use the numbered circle to identify its target.</p><div class="grid" id="grid"></div></main>
<aside class="detail card"><h2 id="slotTitle"></h2><img id="crop" alt="Selected vial position"><label for="classPick">Visible vial class</label><select id="classPick"><option value="">Needs review</option><option>pass</option><option>missing_cap</option><option>crooked_cap</option><option>damaged_vial</option><option>no_vial</option></select><label><input type="checkbox" id="confirmed"> I reviewed this position</label><nav class="position-nav" aria-label="Position navigation"><button type="button" id="previousPosition" aria-label="Previous position">← Previous</button><button type="button" id="nextPosition" aria-label="Next position">Next →</button></nav><p class="fine" id="positionProgress" aria-live="polite"></p><label for="note">Observation or correction</label><textarea id="note"></textarea><p id="evidence" class="fine"></p><p class="fine">Pass means the visible cap and body appear acceptable. An empty hole is “no vial”; it becomes a group “missing vial” only when the fill sequence indicates it should be occupied.</p></aside></div>
<script>
const original=__PAYLOAD__, assets=__ASSETS__, colors=__COLORS__;
let data=structuredClone(original),index=0,area='good',slotId='G01',dirty=false;
const $=id=>document.getElementById(id),key='vial-review-'+data.model_hash+'-'+original.review_revision;
$('snapshotTime').textContent=original.review_timestamp?'Review snapshot: '+original.review_timestamp.replace('T',' ').replace('Z',' UTC'):'';
function validateReview(payload){
 if(!payload||payload.schema_version!==original.schema_version||!Array.isArray(payload.results)||!payload.results.length)throw Error('Invalid review JSON or schema version.');
 const names=new Set(),updates=new Map(),same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
 for(const incoming of payload.results){
  const base=original.results.find(r=>r.filename===incoming.filename);
  if(!base)throw Error('Image not included in this page: '+incoming.filename+'. Open a page containing that image.');
  if(names.has(incoming.filename))throw Error('Duplicate image: '+incoming.filename);names.add(incoming.filename);
  const allowed=[original.model_hash,base.review_provenance?.export_model_hash];
  if(!payload.model_hash||!allowed.includes(payload.model_hash)||incoming.model_hash!==payload.model_hash)throw Error('Model mismatch: '+incoming.filename);
  for(const field of ['source_sha256','image_size','alignment','context'])if(!same(incoming[field],base[field]))throw Error('Image, alignment, or context mismatch: '+incoming.filename);
  if(!Array.isArray(incoming.slots)||incoming.slots.length!==60)throw Error('Expected 60 positions: '+incoming.filename);
  const ids=new Set(),slots=new Map();
  for(const s of incoming.slots){
   const expected=base.slots.find(b=>b.id===s.id);
   if(!expected||ids.has(s.id))throw Error('Invalid or duplicate position.');ids.add(s.id);
   for(const field of ['area','position','row','column_from_right','center','bbox'])if(!same(s[field],expected[field]))throw Error('Position geometry mismatch: '+s.id);
   if(![null,'pass','missing_cap','crooked_cap','damaged_vial','no_vial'].includes(s.class)||!['draft','confirmed'].includes(s.review_status)||typeof s.reason!=='string')throw Error('Invalid class, status, or note: '+s.id);
   if(s.review_status==='confirmed'&&(s.class===null||!base.alignment.accepted))throw Error('Cannot confirm an unresolved or unaligned position: '+s.id);
   if(![true,false,null].includes(s.presence))throw Error('Invalid presence: '+s.id);
   slots.set(s.id,s);
  }
  updates.set(incoming.filename,slots);
 }
 return updates;
}
function applyReview(payload){
 const updates=validateReview(payload),next=structuredClone(data);
 for(const r of next.results){const incoming=updates.get(r.filename);if(!incoming)continue;
  for(const s of r.slots){const incomingSlot=incoming.get(s.id);s.class=incomingSlot.class;s.presence=s.class===null?incomingSlot.presence:s.class!=='no_vial';s.reason=incomingSlot.reason;s.review_status=incomingSlot.review_status;}
  r.annotation_status=r.slots.every(s=>s.review_status==='confirmed')?'reviewed':'draft';
 }
 data=next;return updates.size;
}
try{const saved=JSON.parse(localStorage.getItem(key));if(saved){applyReview(saved);dirty=true}}catch(e){}
const label=c=>c?c.replaceAll('_',' '):'needs review';
function current(){return data.results[index]}function selected(){return current().slots.find(s=>s.id===slotId)}
const positionSequence=data.results.flatMap((r,imageIndex)=>r.slots.map(s=>({imageIndex,id:s.id,area:s.area})));
function positionIndex(){return positionSequence.findIndex(s=>s.imageIndex===index&&s.id===slotId)}
function movePosition(step){const target=positionSequence[positionIndex()+step];if(!target)return;index=target.imageIndex;slotId=target.id;area=target.area;render()}

function save(){dirty=true;try{localStorage.setItem(key,JSON.stringify(data))}catch(e){}$('notice').style.display='block'}
$('loadReview').onclick=()=>$('reviewFile').click();
$('reviewFile').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{
 const count=applyReview(JSON.parse(await file.text()));save();render();$('loadStatus').textContent='Loaded '+count+' images from '+file.name+'. Export JSON to save future edits.';
 }catch(error){$('loadStatus').textContent='Load failed: '+error.message;}finally{e.target.value='';}};
for(let i=0;i<data.results.length;i++){const o=document.createElement('option');o.value=i;o.textContent=data.results[i].filename;$('imagePick').append(o)}
for(const [k,c]of Object.entries(colors)){$('legend').insertAdjacentHTML('beforeend',`<span><i style="background:${c}"></i>${label(k==='null'?null:k)}</span>`)}
function render(){const r=current(),a=assets[r.filename];$('imagePick').value=index;$('areaPick').value=area;
 $('progress').textContent=data.results.reduce((n,r)=>n+r.slots.filter(s=>s.review_status==='confirmed').length,0)+' / '+(data.results.length*60)+' positions reviewed · '+r.slots.filter(s=>s.review_status==='confirmed').length+'/60 in this image';
 const x0=Math.max(0,Math.min(...r.slots.map(s=>s.bbox[0]))-35),y0=Math.max(0,Math.min(...r.slots.map(s=>s.bbox[1]))-35);
 const x1=Math.min(1920,Math.max(...r.slots.map(s=>s.bbox[2]))+35),y1=Math.min(1080,Math.max(...r.slots.map(s=>s.bbox[3]))+35);
 $('scene').setAttribute('viewBox',`${x0} ${y0} ${x1-x0} ${y1-y0}`);
 $('selectionReason').textContent=r.review_selection_reason?'Selected for review: '+r.review_selection_reason:'';
 $('scene').innerHTML=`<image href="${a.scene}" x="0" y="0" width="1920" height="1080"/>`+r.slots.map(s=>{
  const reviewed=s.review_status==='confirmed',status=reviewed?'reviewed':'awaiting review',x=s.center[0],y=s.center[1];
  const badge=reviewed?`<path d="M ${x+13} ${y-20} l 5 5 l 9 -10" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>`:`<text x="${x+20}" y="${y-14}" fill="#101c2d" font-size="20" font-weight="bold" text-anchor="middle">!</text>`;
  return `<g class="svgslot" data-slot="${s.id}" data-review-status="${reviewed?'reviewed':'awaiting'}" role="button" tabindex="0" aria-label="${s.id}, ${label(s.class)}, ${status}"><title>${s.id}: ${label(s.class)} — ${status}</title><circle cx="${x}" cy="${y}" r="24" fill="transparent" stroke="${colors[s.class]}" stroke-width="${s.id===slotId?6:3}" stroke-dasharray="${reviewed?'none':'5 4'}"/><rect x="${x-24}" y="${y+23}" width="52" height="24" fill="#0b1320"/><text x="${x-20}" y="${y+41}" fill="${colors[s.class]}" font-size="18">${s.id}</text><circle class="review-status-badge" cx="${x+20}" cy="${y-20}" r="12" fill="${reviewed?'#16a34a':'#fbbf24'}" stroke="#101c2d" stroke-width="2"/>${badge}</g>`;
 }).join('');
 $('scene').querySelectorAll('[data-slot]').forEach(g=>{const pick=()=>{slotId=g.dataset.slot;area=selected().area;render()};g.onclick=pick;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick()}}});
 $('groups').innerHTML=['good','scrap'].map(g=>{const v=r.groups[g];return `<div class="card"><h2>${g==='good'?'Good tray':'Scrap tray'}</h2><div class="fine">${v.occupied} occupied · ${v.unknown_occupancy} unknown · capacity 30</div>`+Object.entries(v.findings).map(([k,x])=>`<span class="flag ${x===true?'yes':x===null?'unknown':''}">${label(k)}: ${x===null?'unresolved':x?'yes':'no'}</span>`).join('')+`<p class="fine">Missing positions: ${v.missing_positions.join(', ')||'none detected'}<br>Not yet filled: ${v.not_yet_filled_positions.join(', ')||'none'}</p></div>`}).join('');
 $('grid').innerHTML='';const slots=r.slots.filter(s=>s.area===area).sort((a,b)=>a.row-b.row||b.column_from_right-a.column_from_right);
 for(const s of slots){const b=document.createElement('button');b.className='tile'+(s.id===slotId?' active':'');b.style.borderColor=colors[s.class];b.innerHTML=`<img src="${a.crops[s.id]}" alt="${s.id}"><span>${s.id} · ${label(s.class)}</span><span>${s.review_status==='confirmed'?'✓ Reviewed':'! Awaiting review'}</span>`;b.onclick=()=>{slotId=s.id;render()};$('grid').append(b)}
 const s=selected();$('slotTitle').textContent=s.id+' · row '+s.row+', column '+s.column_from_right+' from right';$('crop').src=a.crops[s.id];$('classPick').value=s.class||'';$('confirmed').checked=s.review_status==='confirmed';$('note').value=s.reason;
 const ref=s.reference_match;$('evidence').textContent=ref?`Closest reference: ${ref.image}, ${ref.area} position ${ref.position}. Appearance distance ${ref.distance}; lower is closer. This is not a probability. ${r.alignment.accepted?'Alignment accepted.':'Alignment requires calibration.'}`:'Manual labeling: inspect the marked target in context. No class predictions are supplied.';
 const position=positionIndex();$('previousPosition').disabled=position===0;$('nextPosition').disabled=position===positionSequence.length-1;$('positionProgress').textContent='Position '+(position+1)+' of '+positionSequence.length;
 $('notice').style.display=dirty?'block':'none';
}
$('previousPosition').onclick=()=>movePosition(-1);$('nextPosition').onclick=()=>movePosition(1);
$('imagePick').onchange=e=>{index=+e.target.value;slotId=area==='good'?'G01':'S01';render()};$('areaPick').onchange=e=>{area=e.target.value;slotId=area==='good'?'G01':'S01';render()};
$('classPick').onchange=e=>{const s=selected();s.class=e.target.value||null;s.presence=s.class===null?null:s.class!=='no_vial';s.review_status='draft';save();render()};
$('confirmed').onchange=e=>{const s=selected();if(e.target.checked&&(!s.class||!current().alignment.accepted)){e.target.checked=false;alert('Resolve the visible class and alignment before confirming this position.');return}s.review_status=e.target.checked?'confirmed':'draft';save();render()};
$('note').onchange=e=>{selected().reason=e.target.value;save()};
$('next').onclick=()=>{let n=current().slots.findIndex(s=>s.id===slotId);for(let k=1;k<=60;k++){const s=current().slots[(n+k)%60];if(s.review_status!=='confirmed'){slotId=s.id;area=s.area;render();return}}alert('All positions in this image are reviewed.')};
$('download').onclick=()=>{const exported=structuredClone(data),now=new Date();exported.exported_at=now.toISOString();
 const stamp=exported.exported_at.replace(/:/g,'').replace(/\.\d{3}Z$/,'Z');
 const b=new Blob([JSON.stringify(exported,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='labels-'+stamp+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
render();
</script></html>'''
