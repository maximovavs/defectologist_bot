"""R3A R2 H1 independent offline mutation probes (bounded, no external services)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = (ROOT / 'research/r3a_r2_final_post_contract.py').read_text(encoding='utf-8')
MUTANTS = {
    'M01_allow_without_simulation': ('if simulation_only is not True:', 'if False:'),
    'M02_skip_source_url_registry': ('if type(registered_url) is not str or registered_url != source.canonical_url:', 'if False:'),
    'M03_skip_final_text_digest': ('if compare_digest(record["approved_final_post_sha256"], post_hash):', 'if True:'),
    'M04_skip_signature_verify': (
        'Ed25519PublicKey.from_public_bytes(trusted_key_bytes).verify(\n            reviewed.signature, signed_bytes)',
        'pass # test-only mutation: signature verification disabled'),
    'M05_skip_raw_html_sha': (
        '(permit["raw_html_sha256"], digest(source.raw_response_bytes)),',
        '(permit["raw_html_sha256"], permit["raw_html_sha256"]),'),
    'M06_skip_evidence_sha': (
        '(permit["evidence_sha256"], evidence_hash),',
        '(permit["evidence_sha256"], permit["evidence_sha256"]),'),
    'M07_skip_topic': ('(permit["topic_id"], source.topic_id),', '(permit["topic_id"], permit["topic_id"]),'),
    'M08_skip_population': ('(permit["population"], source.population),', '(permit["population"], permit["population"]),'),
    'M09_skip_age_scope': ('(permit["age_scope"], source.age_scope),', '(permit["age_scope"], permit["age_scope"]),'),
    'M10_skip_full_refutation': (
        'if not _full_refutation_sentence(source.evidence_text, Span(ref["start"], ref["text"])):',
        'if False:'),
    'M11_skip_source_qualifiers': (
        'if (not qualifiers or any(not _full_refutation_sentence(\n            source.evidence_text, Span(item["start"], item["text"]))\n            for item in qualifiers)):',
        'if False:'),
    'M12_skip_post_qualifiers': (
        'if (not permit["mandatory_post_qualifiers"] or any(\n            not q or q not in permit["fact_text"]\n            for q in permit["mandatory_post_qualifiers"])):',
        'if False:'),
    'M13_skip_fact_section': (
        'if not permit["fact_text"] or fact_area.count(permit["fact_text"]) != 1:',
        'if False:'),
    'M14_skip_post_age': (
        'if len(ages) != 1 or ages[0].group(1).strip() != permit["age_scope"]:',
        'if False:'),
    'M15_skip_myth_line': (
        'if len(myth_matches) != 1 or myth_matches[0].group(1).strip() != permit["myth_line"]:',
        'if False:'),
    'M16_allow_permit_subtypes': ('if type(permit) is not ReviewPermit:', 'if False:'),
    'M17_allow_str_subtypes': (
        'if any(type(getattr(permit, key)) is not str for key in _SIGNED_STRING_FIELDS):',
        'if False:'),
    'M18_allow_signed_review_subtypes': (
        'if type(signed) is not SignedReview or type(signed.signature) is not bytes:',
        'if False:'),
    'M19_allow_source_subtypes': ('if type(source) is not SourceSnapshot:', 'if False:'),
    'M20_allow_span_subtypes': ('if not _strict_span(permit.refutation):', 'if False:'),
    'M21_bypass_prime_subgroup_gate': (
        'if not _ed25519_prime_order_public_key(trusted_key_bytes):',
        'if False:'),
    'M22_allow_noncanonical_y': (
        'if y >= p:',
        'if False:'),
    'M23_allow_non_subgroup_points': (
        'return ax == 0 and (ay - az) % p == 0 and at == 0',
        'return True'),
}
results=[]
for mutant, (old,new) in MUTANTS.items():
    count=ORIGINAL.count(old)
    if count!=1:
        raise SystemExit(f'Mutation anchor mismatch {mutant}: {count}')
    with tempfile.TemporaryDirectory(prefix='r3a_r2_security_mut_') as folder:
        root=Path(folder)
        (root/'research').mkdir()
        (root/'tests').mkdir()
        (root/'evidence').mkdir()
        (root/'research/r3a_r2_final_post_contract.py').write_text(ORIGINAL.replace(old,new),encoding='utf-8')
        for path in (ROOT/'tests').glob('test_r3a_r2*.py'):
            shutil.copy2(path,root/'tests'/path.name)
        shutil.copy2(ROOT/'evidence/source_admission_20261027_historical_UNVERIFIED.json',root/'evidence/source_admission_20261027_historical_UNVERIFIED.json')
        env=os.environ.copy()
        env.update({'PYTHONPATH':str(root),'PYTHONNOUSERSITE':'1','HTTP_PROXY':'http://127.0.0.1:9','HTTPS_PROXY':'http://127.0.0.1:9','ALL_PROXY':'http://127.0.0.1:9'})
        p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_r3a_r2*.py'],cwd=root,env=env,text=True,capture_output=True,timeout=20)
        results.append({'mutation':mutant,'detected':p.returncode!=0,'test_exit_code':p.returncode,'test_output_tail':p.stderr[-180:]})
summary={'python':sys.version.split()[0],'count':len(results),'detected':sum(x['detected'] for x in results),'results':results}
print(json.dumps(summary,indent=2))
raise SystemExit(0 if summary['count']==summary['detected'] else 1)
