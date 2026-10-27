"""R6-D entirely disconnected local proof: review status is NOT a signature."""
import json
import hashlib
from dataclasses import replace
from pathlib import Path
import unittest
from src.services import safe03_r6d_offline_acceptance as a

FROZEN=Path(__file__).resolve().parents[1]/'SAFE03_R6D_FROZEN_CANDIDATES.json'
MANIFEST=json.loads(FROZEN.read_text(encoding='utf-8'))
FAKE_FIXTURE_REGISTRY={cid:a.digest(entry) for cid,entry in a.CATALOG.items()}


def proposed(entry, age=None):
    return a.proposal(entry, entry.age_min_months if age is None else age)


def test_freeze_exact_text_hashes_full_digests_all_fields():
    assert len(MANIFEST['entries'])==len(a.CATALOG)==5
    for record in MANIFEST['entries']:
        e=a.CATALOG[record['catalog_id']]
        assert record['version']==e.version
        assert record['text']==e.text
        assert hashlib.sha256(e.text.encode()).hexdigest()==record['text_sha256']
        assert a.digest(e)==record['entry_sha256']
        assert record['metadata_approval']=='PENDING_VERIFIED_METADATA_APPROVAL'
        assert record['text_adjudication']=='YES_USER_RELAYED_NOT_INDEPENDENTLY_AUTHENTICATED'


def test_public_registry_empty_and_no_auto_yes():
    assert not a.PRODUCTION_APPROVAL_DIGESTS
    for e in a.CATALOG.values():
        p=proposed(e)
        assert a.render(p).decision=='NO_POST'
        simulated=a._test_fixture_only_render(p,FAKE_FIXTURE_REGISTRY)
        assert simulated.decision=='ALLOW'
        assert a.validate_final(simulated,simulated.text,simulated.lane).decision=='NO_POST'


def test_simulated_exact_roundtrip_both_boundary_ages():
    for e in a.CATALOG.values():
        for age in [e.age_min_months,e.age_max_months if e.age_max_months is not None else 240]:
            p=proposed(e,age)
            result=a._test_fixture_only_render(p,FAKE_FIXTURE_REGISTRY)
            assert result.decision=='ALLOW'
            assert result.lane==('safety_warning' if e.kind=='warning' else 'exercise')
            assert result.text==e.text and result.recipient==e.recipient
            assert a._test_fixture_only_validate(result,result.text,result.lane,FAKE_FIXTURE_REGISTRY).decision=='ALLOW'


def test_warning_age_bound_corrected_and_exercises_still_bounded():
    for cid in ('warning_button_batteries','warning_small_magnets'):
        e=a.CATALOG[cid]
        assert e.age_max_months is None and e.age_min_months==0
        for age in (0,12,216,217,600,1200):
            assert a._test_fixture_only_render(proposed(e,age),FAKE_FIXTURE_REGISTRY).decision=='ALLOW'
        for age in (-1,None,False,True,2.5,'216'):
            p=proposed(e); p['age_months']=age
            assert a._test_fixture_only_render(p,FAKE_FIXTURE_REGISTRY).decision=='NO_POST'
    for cid in ('picture_naming_24_48','soft_ball_roll_36_60','clap_rhythm_24_48'):
        e=a.CATALOG[cid]
        assert e.age_max_months is not None
        for age in (-1,e.age_min_months-1,e.age_max_months+1):
            assert a._test_fixture_only_render(proposed(e,age),FAKE_FIXTURE_REGISTRY).decision=='NO_POST'


def test_no_drift_in_core_fields():
    for field, bad in (('text', 'new text'), ('version', 999), ('kind', 'warning'),
                       ('object_id', 'button_battery'), ('action', 'unsafe'),
                       ('actor', 'unsafe'), ('recipient', 'child'),
                       ('constraints', ['unreviewed'])):
        for e in a.CATALOG.values():
            p=proposed(e)
            if p[field]==bad:
                continue
            p[field]=bad
            assert a._test_fixture_only_render(p,FAKE_FIXTURE_REGISTRY).decision=='NO_POST', (field, e.catalog_id)


def test_untrusted_flags_and_missing_keys_and_unknown_ids():
    for e in a.CATALOG.values():
        p=proposed(e)
        for addition in ({'approved':True},{'admin':True},{'review_digest':a.digest(e)}):
            altered={**p,**addition}
            assert a._test_fixture_only_render(altered,FAKE_FIXTURE_REGISTRY).decision=='NO_POST'
        for field in list(p):
            altered=dict(p); altered.pop(field)
            assert a._test_fixture_only_render(altered,FAKE_FIXTURE_REGISTRY).decision=='NO_POST'
        altered=dict(p); altered['catalog_id']='wrong_id'
        assert a._test_fixture_only_render(altered,FAKE_FIXTURE_REGISTRY).decision=='NO_POST'


def test_empty_wrong_stale_and_old_catalog_digest_reject():
    old_digests={'warning_button_batteries':'ca3c4adea1953939b0fb3045cf3b6fdaec0e9beefa2dbaadb74e16e4819c8564',
                 'warning_small_magnets':'292bc960fca59025f049546b2246aa58fa2ddf261fe6e59babb4f42495055ebe'}
    for e in a.CATALOG.values():
        for approvals in ({},{e.catalog_id:'0'*64},{e.catalog_id:a.digest(replace(e,text=e.text+'x'))}):
            assert a._test_fixture_only_render(proposed(e),approvals).decision=='NO_POST'
    for cid,old in old_digests.items():
        assert old != a.digest(a.CATALOG[cid])
        assert a._test_fixture_only_render(proposed(a.CATALOG[cid]),{cid:old}).decision=='NO_POST'


def test_post_render_all_frozen_fields_and_warning_route():
    for e in a.CATALOG.values():
        p=proposed(e)
        out=a._test_fixture_only_render(p,FAKE_FIXTURE_REGISTRY)
        for field,bad in [('text',e.text+'!'),('version',-1),('age_months',-1),('object_id','coin'),('action','changed'),
                          ('actor','other'),('recipient','other'),('constraints',('not_approved',)),('digest','0'*64),
                          ('seal','0'*64),('lane','exercise' if out.lane=='safety_warning' else 'safety_warning')]:
            mod=replace(out,**{field:bad})
            assert a._test_fixture_only_validate(mod,mod.text,mod.lane,FAKE_FIXTURE_REGISTRY).decision=='NO_POST',(e.catalog_id,field)
        assert a._test_fixture_only_validate(out,e.text,'exercise' if out.lane=='safety_warning' else 'safety_warning',FAKE_FIXTURE_REGISTRY).decision=='NO_POST'
        for corrupt in (e.text+'\n',e.text+' ',e.text.replace(' ','\u00a0',1),'⚠️\n'+e.text,e.text+' Добавим игру с батарейкой.'):
            assert a._test_fixture_only_validate(out,corrupt,out.lane,FAKE_FIXTURE_REGISTRY).decision=='NO_POST'


def test_metadata_change_alters_digest_even_with_same_text():
    for e in a.CATALOG.values():
        assert a.digest(e)!=a.digest(replace(e,age_min_months=e.age_min_months+1))
        assert a.digest(e)!=a.digest(replace(e,recipient='different'))
        assert a.digest(e)!=a.digest(replace(e,object_id='different'))
    for cid in ('warning_small_magnets','warning_button_batteries'):
        e=a.CATALOG[cid]
        assert a.digest(e)!=a.digest(replace(e,age_max_months=216))


def test_superseded_legacy_modules_not_packaged_and_old_text_not_approved():
    import importlib.util
    # Do not ship obsolete v1 renderer or any executable v1 catalogue.
    assert importlib.util.find_spec('src.services.safe03_curated_boundary') is None
    assert importlib.util.find_spec('src.services.safe03_r6b_offline_renderer') is None
    old_fragments = (
        'Попросите указать на знакомый предмет и назовите его вместе.',
        'Взрослый хлопает в ладоши два раза и делает паузу.',
        'Не используйте кнопочные батарейки в играх с детьми. Храните их вне доступа ребёнка.',
    )
    assert all(old != entry.text for entry in a.CATALOG.values() for old in old_fragments)
    for entry in a.CATALOG.values():
        p = proposed(entry)
        p['text'] = 'Не используйте кнопочные батарейки в играх с детьми. Храните их вне доступа ребёнка.'
        assert a._test_fixture_only_render(p,FAKE_FIXTURE_REGISTRY).decision == 'NO_POST'


def test_toctou_flip_dict_subclass_never_admitted():
    class Flip(dict):
        reads = 0
        def __getitem__(self, key):
            if key == 'age_months':
                self.reads += 1
                return 30 if self.reads == 1 else 999
            return super().__getitem__(key)
    raw = Flip(proposed(a.CATALOG['picture_naming_24_48'], 30))
    result = a._test_fixture_only_render(raw, FAKE_FIXTURE_REGISTRY)
    assert result.decision == 'NO_POST'
    assert result.reason == 'invalid_fields'
    assert raw.reads == 0


def test_snapshot_captures_validated_age_and_final_invariant():
    entry = a.CATALOG['picture_naming_24_48']
    raw = proposed(entry, 30)
    result = a._test_fixture_only_render(raw, FAKE_FIXTURE_REGISTRY)
    raw['age_months'] = 999
    raw['text'] = 'tampered'
    assert result.decision == 'ALLOW' and result.age_months == 30
    assert a._test_fixture_only_validate(result, result.text, result.lane,FAKE_FIXTURE_REGISTRY).decision == 'ALLOW'
    assert a._test_fixture_only_render(raw,FAKE_FIXTURE_REGISTRY).decision == 'NO_POST'


def test_dataclass_subclass_rejected_in_final_validator():
    entry=a.CATALOG['warning_button_batteries']
    out=a._test_fixture_only_render(proposed(entry,12), FAKE_FIXTURE_REGISTRY)
    class Subtype(a.Rendered):
        pass
    import dataclasses
    sub=Subtype(**dataclasses.asdict(out))
    assert a._test_fixture_only_validate(sub,sub.text,sub.lane,FAKE_FIXTURE_REGISTRY).decision=='NO_POST'


def test_legacy_unstructured_and_none_rejected():
    for x in [None,False,[],{},'Текст',{'text':'Катаем мяч'}, {'age_months':12,'approved':True}]:
        assert a.render(x).decision=='NO_POST'
        assert a._test_fixture_only_render(x,FAKE_FIXTURE_REGISTRY).decision=='NO_POST'


def load_tests(loader, tests, pattern):
    """Run all R6-D proof functions under the SAFE03 unittest-only CI runner."""
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
