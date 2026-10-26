import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import labeler
from rules import summarize


class ParticipantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results,cls.sources=labeler.prepare([labeler.HERE/'examples/tray_000001.jpg'],
                                               profile_id='view_001',offset=(0,0))

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.output=self.root/'preview'
        labeler.save(self.results,self.sources,self.output)
        self.template=labeler.read(self.output/'labels.json')

    def tearDown(self):
        self.temp.cleanup()

    def test_all_positions_unresolved_with_portable_sources(self):
        r=self.template['results'][0]
        self.assertEqual(len(r['slots']),60)
        self.assertTrue(all(s['class'] is None and s['presence'] is None and s['review_status']=='draft' for s in r['slots']))
        self.assertFalse(Path(r['source_path']).is_absolute())
        self.assertEqual(labeler.sha(self.output/r['source_path']),r['source_sha256'])
        self.assertEqual(len(list((self.output/'tray_000001/crops').glob('*.jpg'))),60)
        self.assertIsNone(r['groups']['scrap']['findings']['good_misclassified_as_bad'])

    def test_review_roundtrip_recomputes_presence_and_groups(self):
        payload=copy.deepcopy(self.template)
        for s in payload['results'][0]['slots']:
            s.update({'class':'pass','review_status':'confirmed','presence':False})
        labeler.write(self.root/'review.json',payload)
        labeler.import_review(self.root/'review.json',self.output/'labels.json',self.root/'reviewed')
        r=labeler.read(self.root/'reviewed/labels.json')['results'][0]
        self.assertTrue(r['groups']['good']['findings']['complete_order'])
        self.assertTrue(all(s['presence'] is True for s in r['slots']))
        self.assertIsNone(r['groups']['scrap']['findings']['good_misclassified_as_bad'])

    def test_geometry_and_metadata_tampering_rejected(self):
        mutations=[lambda p:p['results'][0]['slots'][0]['bbox'].__setitem__(0,0),
                   lambda p:p['results'][0].__setitem__('source_path','../../outside.jpg'),
                   lambda p:p['results'][0].__setitem__('source_sha256','changed'),
                   lambda p:p['results'][0]['slots'][1].__setitem__('id','G01'),
                   lambda p:p['results'][0]['slots'][0].__setitem__('class','invalid'),
                   lambda p:p['results'][0]['slots'][0].__setitem__('review_status','confirmed'),
                   lambda p:p.__setitem__('model_hash','changed')]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                payload=copy.deepcopy(self.template);mutate(payload)
                with self.assertRaises(ValueError):labeler.merge_review(payload,self.template)

    def test_source_change_rejected_before_output_created(self):
        image=self.output/self.template['results'][0]['source_path']
        image.write_bytes(b'changed')
        labeler.write(self.root/'review.json',self.template)
        with self.assertRaisesRegex(ValueError,'Source image changed'):
            labeler.import_review(self.root/'review.json',self.output/'labels.json',self.root/'reviewed')
        self.assertFalse((self.root/'reviewed').exists())

    def test_moved_review_folder_still_imports(self):
        moved=self.root/'moved'
        shutil.move(self.output,moved)
        labeler.write(self.root/'review.json',self.template)
        labeler.import_review(self.root/'review.json',moved/'labels.json',self.root/'reviewed')
        self.assertTrue((self.root/'reviewed/index.html').exists())

    def test_existing_review_not_overwritten(self):
        before=labeler.sha(self.output/'labels.json')
        with self.assertRaisesRegex(ValueError,'already exists'):
            labeler.save(self.results,self.sources,self.output)
        self.assertEqual(labeler.sha(self.output/'labels.json'),before)

    def test_public_reference_alignment_and_wrong_size(self):
        results,_=labeler.prepare([labeler.HERE/'examples/tray_000001.jpg'])
        self.assertEqual(results[0]['alignment']['offset'],[0,0])
        from PIL import Image
        path=self.root/'small.png';Image.new('RGB',(640,480)).save(path)
        with self.assertRaisesRegex(ValueError,'1920x1080'):
            labeler.prepare([path],profile_id='view_001',offset=(0,0))

    def test_duplicate_stems_rejected(self):
        with self.assertRaisesRegex(ValueError,'unique filenames'):
            labeler.prepare([self.root/'one.jpg',self.root/'one.png'],profile_id='view_001',offset=(0,0))

    def test_subset_review_preserves_other_images(self):
        template=copy.deepcopy(self.template)
        other=copy.deepcopy(template['results'][0]);other['filename']='another.jpg'
        template['results'].append(other)
        payload=copy.deepcopy(self.template)
        payload['results'][0]['slots'][0].update({'class':'missing_cap','review_status':'confirmed'})
        merged=labeler.merge_review(payload,template)
        self.assertEqual(merged[0]['slots'][0]['class'],'missing_cap')
        self.assertEqual(merged[1],other)

    def test_notes_cannot_break_embedded_script_or_template_tokens(self):
        import re
        from report import build_report
        results=copy.deepcopy(self.template['results'])
        note='</script>__ASSETS__ <script>example</script>'
        results[0]['slots'][0]['reason']=note
        build_report(results,self.output/'special.html')
        html=(self.output/'special.html').read_text()
        self.assertNotIn('</script>__ASSETS__',html)
        payload=json.loads(re.search(r'const original=(.*), assets=',html).group(1))
        self.assertEqual(payload['results'][0]['slots'][0]['reason'],note)

    def test_fill_order_distinguishes_internal_gap_and_trailing_empty(self):
        slots=[dict(position=p,presence=p not in (2,30),**{'class':'no_vial' if p in (2,30) else 'pass'},review_status='confirmed') for p in range(1,31)]
        result=summarize('good',slots)
        self.assertEqual(result['missing_positions'],[2])
        self.assertEqual(result['not_yet_filled_positions'],[30])


if __name__=='__main__':
    unittest.main()
