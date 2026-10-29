"""Offline mutation coverage for fabricated send payload and unsigned permits."""
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
sys.path.insert(0,str(ROOT/'research'))
sys.path.insert(0,str(ROOT/'reference'))
from test_r3a_r3_approved_card_contract import R3OfflineWholeCard
from r3a_r3_approved_card_contract import SignedPack, evaluate_last_boundary

x=R3OfflineWholeCard(); x.setUp()
accepted=x.run_decision(card_id='a')
assert accepted.allowed
base=accepted.send_plan['text']; text_attempts=0; text_false_accept=0
rnd=random.Random(20261029)
for _ in range(1000):
    i=rnd.randrange(len(base)); character=rnd.choice(['.', '!', 'X', '9', '\u200b'])
    if character == base[i]: character='Z'
    altered=base[:i]+character+base[i+1:]
    outcome=x.run_decision(proposed_transport={'method':'sendMessage','text':altered,
        'parse_mode':None,'disable_web_page_preview':True})
    text_attempts+=1
    if outcome.allowed:text_false_accept+=1
assert text_false_accept==0
pack=x.sign(); payload=pack.payload
payload_attempts=0; signed_false_accept=0
for _ in range(512):
    buf=bytearray(payload)
    at=rnd.randrange(len(buf)); buf[at]^=1<<rnd.randrange(8)
    altered=SignedPack(bytes(buf),pack.signature)
    d=x.run_decision(signed_pack=altered)
    payload_attempts+=1
    if d.allowed:signed_false_accept+=1
assert signed_false_accept==0
assert not evaluate_last_boundary(signed_pack=pack,source=x.source,authority=x.auth,card_id='a').allowed
print(f"offline probe: {text_attempts} final-text mutations, false_ACCEPT={text_false_accept}; "
      f"{payload_attempts} signed-payload mutations, false_ACCEPT={signed_false_accept}; "
      f"default production admission=DENY")
