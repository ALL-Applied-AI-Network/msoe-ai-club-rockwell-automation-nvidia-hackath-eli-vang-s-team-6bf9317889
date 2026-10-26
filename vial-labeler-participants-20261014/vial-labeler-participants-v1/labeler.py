#!/usr/bin/env python3
"""Offline manual vial annotation: prepare a review page and import reviewed JSON."""
import argparse
import copy
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageOps

from rules import CLASSES, summarize

HERE = Path(__file__).resolve().parent
VERSION = 'participant-1.0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def center(grid, position):
    column, row = divmod(position-1, 5)
    u, v = 1-column/5, row/4
    tl, tr, bl, br = np.asarray(grid, dtype=float)
    x, y = (1-v)*((1-u)*tl+u*tr)+v*((1-u)*bl+u*br)
    return float(x), float(y)


def anchor(im, boxes, offset):
    dx, dy = offset
    values = []
    for left, top, right, bottom in boxes:
        crop = im.crop((left+dx,top+dy,right+dx,bottom+dy)).convert('L').resize((48,24))
        values.append(np.asarray(crop,dtype=float).ravel()/255)
    values = np.concatenate(values)
    return (values-values.mean())/max(values.std(),.05)


def fingerprint(config):
    parts = [sha(HERE/name) for name in ('labeler.py','rules.py')]
    return hashlib.sha256((json.dumps(config,sort_keys=True)+''.join(parts)).encode()).hexdigest()


def prepare(paths, config_path=HERE/'calibration.json', profile_id=None, offset=None):
    config = read(config_path)
    if config['image_size'] != [1920,1080]:
        raise ValueError('This review layout supports 1920x1080 images only; recalibration and layout changes are required')
    profiles = config['profiles']
    if profile_id:
        profiles = [p for p in profiles if p['id']==profile_id]
        if not profiles:
            raise ValueError('Unknown calibration profile')
    if offset is not None and profile_id is None:
        raise ValueError('--offset requires an explicitly selected --profile')
    paths = [Path(p).resolve() for p in paths]
    if not paths or len({p.name for p in paths})!=len(paths) or len({p.stem for p in paths})!=len(paths):
        raise ValueError('Select at least one image with unique filenames and stems')
    references = {}
    if offset is None:
        for p in profiles:
            reference = Path(config_path).resolve().parent/'examples'/p['reference_image']
            if sha(reference)!=p['reference_sha256']:
                raise ValueError('Calibration reference bytes changed')
            references[p['id']] = anchor(image(reference),p['alignment_anchors'],(0,0))
    results = []
    for path in paths:
        im = image(path)
        if im.size != (1920,1080):
            raise ValueError(f'{path.name}: expected 1920x1080, got {im.size}; changed views need recalibration')
        if offset is None:
            candidates = []
            for p in profiles:
                for dy in range(-18,19,3):
                    for dx in range(-18,19,3):
                        residual = float(np.mean((anchor(im,p['alignment_anchors'],(dx,dy))-references[p['id']])**2))
                        candidates.append((residual,p['id'],dx,dy))
            residual, selected, dx, dy = min(candidates)
            if residual > config['max_alignment_error'] or max(abs(dx),abs(dy))>=18:
                raise ValueError(f'{path.name}: unfamiliar fixture alignment; inspect geometry and select --profile/--offset explicitly')
        else:
            selected, (dx,dy), residual = profile_id, offset, None
        p = next(p for p in profiles if p['id']==selected)
        result = dict(schema_version=VERSION,filename=path.name,source_path='images/'+path.name,
            source_sha256=sha(path),image_size=list(im.size),model_hash=fingerprint(config),
            backend='manual_annotation',annotation_status='draft',
            alignment=dict(profile=selected,offset=[dx,dy],residual=residual,accepted=True,
                           geometry_requires_visual_check=True),
            context={},slots=[],groups={})
        for area,prefix in (('good','G'),('scrap','S')):
            for position in range(1,31):
                x,y = center(p['areas'][area]['grid'],position)
                x,y = x+dx,y+dy
                left,top,right,bottom = p['areas'][area]['crop_offsets']
                box = [round(x+left),round(y+top),round(x+right),round(y+bottom)]
                if not (0<=box[0]<box[2]<=1920 and 0<=box[1]<box[3]<=1080):
                    raise ValueError('Calibrated crop extends outside image')
                result['slots'].append(dict(id=f'{prefix}{position:02}',area=area,position=position,
                    row=(position-1)%5+1,column_from_right=(position-1)//5+1,
                    center=[round(x,2),round(y,2)],bbox=box,
                    **{'class':None},presence=None,review_status='draft',
                    reason='',reference_match=None,suggested_class=None))
        recompute(result)
        results.append(result)
    return results, {p.name:p for p in paths}


def recompute(result):
    result['groups'] = {area:summarize(area,[s for s in result['slots'] if s['area']==area])
                        for area in ('good','scrap')}
    result['annotation_status'] = 'reviewed' if all(s['review_status']=='confirmed' for s in result['slots']) else 'draft'


def save(results, sources, output):
    """Write portable copies; never replace a previous review directory."""
    from report import build_report, overlay, contact_sheet
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a new directory')
    for r in results:
        if sha(sources[r['filename']])!=r['source_sha256']:
            raise ValueError('Source image changed')
    output.mkdir(parents=True)
    (output/'images').mkdir()
    results = copy.deepcopy(results)
    for r in results:
        shutil.copyfile(sources[r['filename']],output/r['source_path'])
        im = image(output/r['source_path'])
        folder = output/Path(r['filename']).stem
        (folder/'crops').mkdir(parents=True)
        for s in r['slots']:
            s['crop_path'] = folder.name+'/crops/'+s['id']+'.jpg'
            im.crop(s['bbox']).save(output/s['crop_path'],quality=95)
        write(folder/'labels.json',r)
        overlay(im,r).save(folder/'annotated.jpg',quality=95)
        for area in ('good','scrap'):
            contact_sheet(im,r,area).save(folder/(area+'-positions.jpg'),quality=95)
    write(output/'labels.json',dict(schema_version=VERSION,model_hash=results[0]['model_hash'],results=results))
    with (output/'slots.csv').open('w',newline='',encoding='utf-8') as handle:
        fields=['image','slot','area','position','class','presence','review_status','reason']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in results:
            for s in r['slots']:
                writer.writerow(dict(image=r['filename'],slot=s['id'],**{k:s[k] for k in fields[2:]}))
    write(output/'manifest.json',dict(version=VERSION,mode='manual',created_at=datetime.now(timezone.utc).isoformat(),
        images=[dict(filename=r['filename'],source_sha256=r['source_sha256']) for r in results]))
    build_report(results,output/'index.html')
    print(f'{len(results)} images, {60*len(results)} positions. Open {output/"index.html"}')


def merge_review(payload, template):
    """Only editable label fields may change; preserve canonical geometry and sources."""
    if payload.get('schema_version')!=VERSION or payload.get('model_hash')!=template['model_hash']:
        raise ValueError('Review version or tool/calibration mismatch')
    base = copy.deepcopy(template['results'])
    known = {r['filename']:r for r in base}
    if not payload.get('results'):
        raise ValueError('Review contains no images')
    seen = set()
    for incoming in payload['results']:
        name = incoming['filename']
        if name not in known or name in seen:
            raise ValueError('Unknown or duplicate image')
        seen.add(name);r=known[name]
        for field in ('source_path','source_sha256','image_size','model_hash','alignment','context'):
            if incoming.get(field)!=r.get(field):
                raise ValueError('Immutable image metadata changed: '+field)
        if len(incoming['slots'])!=60:
            raise ValueError('Exactly 60 positions required')
        slots = {s['id']:s for s in r['slots']};seen_slots=set()
        for s in incoming['slots']:
            if s['id'] not in slots or s['id'] in seen_slots:
                raise ValueError('Invalid or duplicate slot')
            seen_slots.add(s['id']);original=slots[s['id']]
            for field in ('area','position','row','column_from_right','center','bbox','crop_path'):
                if s.get(field)!=original.get(field):
                    raise ValueError('Immutable position geometry changed: '+field)
            if s.get('class') not in CLASSES+[None] or s.get('review_status') not in ('draft','confirmed'):
                raise ValueError('Invalid class or review state')
            if s['review_status']=='confirmed' and s['class'] is None:
                raise ValueError('Cannot confirm an unresolved class')
            if not isinstance(s.get('reason'),str) or s.get('presence') is not None and type(s.get('presence')) is not bool:
                raise ValueError('Invalid note or presence')
            original.update({k:s[k] for k in ('class','review_status','reason')})
            original['presence'] = None if s['class'] is None else s['class']!='no_vial'
        recompute(r)
    return base


def import_review(review_path, template_path, output):
    template_path=Path(template_path).resolve()
    results=merge_review(read(review_path),read(template_path))
    sources={}
    for r in results:
        path=(template_path.parent/r['source_path']).resolve()
        if not path.is_relative_to(template_path.parent) or Path(r['source_path']).is_absolute():
            raise ValueError('Template source path escapes review directory')
        sources[r['filename']]=path
    save(results,sources,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run',help='Prepare an unresolved manual review')
    run.add_argument('--images',type=Path,default=HERE/'examples')
    run.add_argument('--files',nargs='+',default=['tray_000001.jpg'])
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--calibration',type=Path,default=HERE/'calibration.json')
    run.add_argument('--profile',help='Explicitly select a geometry profile')
    run.add_argument('--offset',type=int,nargs=2,metavar=('DX','DY'),help='Verified manual translation; requires --profile')
    review=sub.add_parser('import-review',help='Validate browser export against the original portable template')
    review.add_argument('review',type=Path)
    review.add_argument('--template',type=Path,required=True,help='Original review folder labels.json')
    review.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=='run':
            if any(Path(name).name!=name for name in args.files):
                raise ValueError('--files accepts basenames only')
            results,sources=prepare([args.images/name for name in args.files],args.calibration,args.profile,args.offset)
            save(results,sources,args.output)
        else:
            import_review(args.review,args.template,args.output)
    except (ValueError,KeyError,TypeError,OSError) as error:
        parser.exit(2,f'Error: {error}\n')


if __name__=='__main__':
    main()
