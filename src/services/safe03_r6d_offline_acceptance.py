"""SAFE03 R6-D, isolated offline acceptance candidate. NEVER import in publisher.

Exact USER-RELAYED clinician-adjudicated texts; no independently authenticated
identity, signed whole-entry digest, age metadata approval, or production release.
Production-facing approval registry is intentionally empty. Test-only fixture
approvals do not imply specialist approval and must never be wired to production.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from types import MappingProxyType
from typing import Mapping
import hashlib
import hmac
import json
import secrets


@dataclass(frozen=True)
class Entry:
    catalog_id: str
    version: int
    kind: str
    age_min_months: int
    age_max_months: int | None  # None: no upper limit; medical metadata still UNAPPROVED
    object_id: str
    action: str
    actor: str
    recipient: str
    constraints: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Decision:
    decision: str
    reason: str
    catalog_id: str = ""
    age_months: int | None = None  # value captured at successful admission, never re-read


@dataclass(frozen=True)
class Rendered:
    decision: str
    reason: str
    lane: str = "none"
    catalog_id: str = ""
    version: int = 0
    age_months: int = -1
    object_id: str = ""
    action: str = ""
    actor: str = ""
    recipient: str = ""
    constraints: tuple[str, ...] = ()
    text: str = ""
    digest: str = ""
    seal: str = ""


_ENTRIES = (
    Entry(
        catalog_id='picture_naming_24_48',
        version=2,
        kind='exercise',
        age_min_months=24,
        age_max_months=48,
        object_id='large_picture_card',
        action='point_to_picture',
        actor='child',
        recipient='adult_caregiver',
        constraints=('adult_present', 'no_detachable_small_parts'),
        text='Покажите ребёнку крупную картинку без съёмных деталей. Предложите указать на знакомый предмет и назовите его вместе.\n\nОтвет жестом — полноценный способ общения. Поддерживайте любые попытки взаимодействия, не требуйте обязательного произнесения слова. Если у ребёнка есть особенности зрения или моторики, адаптируйте материал: используйте подходящие контрастные изображения и удобное расположение предметов.',
    ),
    Entry(
        catalog_id='soft_ball_roll_36_60',
        version=2,
        kind='exercise',
        age_min_months=36,
        age_max_months=60,
        object_id='large_soft_ball',
        action='roll_ball',
        actor='child',
        recipient='adult_caregiver',
        constraints=('adult_present', 'no_detachable_small_parts'),
        text='Взрослый и ребёнок по очереди катят друг другу крупный мягкий мяч без съёмных мелких деталей.\n\nИгра добровольная. Подбирайте расстояние и движение с учётом возможностей ребёнка. Наша цель — учиться взаимодействовать по очереди, а не оценивать спортивные способности. Не требуйте обязательного взгляда в глаза.',
    ),
    Entry(
        catalog_id='clap_rhythm_24_48',
        version=2,
        kind='exercise',
        age_min_months=24,
        age_max_months=48,
        object_id='hands',
        action='clap_rhythm',
        actor='adult_caregiver',
        recipient='child_observer',
        constraints=('adult_present', 'stop_if_distressed'),
        text='Взрослый негромко хлопает в ладоши два раза и делает паузу. Предложите ребёнку повторить, если он хочет.\n\nНе хлопайте резко или рядом с ушами ребёнка. Прекратите игру при дискомфорте. Это упражнение не является проверкой слуха. Если ребёнок регулярно слабо реагирует на обычные звуки, необходимо обратиться к сурдологу для оценки слуха.',
    ),
    Entry(
        catalog_id='warning_small_magnets',
        version=2,
        kind='warning',
        age_min_months=0,
        age_max_months=None,
        object_id='small_magnet',
        action='warn_no_child_access',
        actor='adult_caregiver',
        recipient='adult_caregiver',
        constraints=('no_child_access', 'no_handling_exercise'),
        text='Не используйте мелкие магниты для речевых игр с детьми. Храните их вне доступа ребёнка.\n\nПроглатывание нескольких магнитов или магнита вместе с металлическим предметом может привести к тяжёлому повреждению кишечника, в том числе со смертельным исходом.\n\nПри подозрении на проглатывание магнита срочно обращайтесь за медицинской помощью, даже если ребёнок чувствует себя нормально. Врач определит необходимость рентгенологического обследования. Не ждите появления симптомов.',
    ),
    Entry(
        catalog_id='warning_button_batteries',
        version=3,
        kind='warning',
        age_min_months=0,
        age_max_months=None,
        object_id='button_battery',
        action='warn_no_child_access',
        actor='adult_caregiver',
        recipient='adult_caregiver',
        constraints=('no_child_access', 'no_handling_exercise'),
        text='Не используйте кнопочные батарейки в играх с детьми. Храните их вне доступа ребёнка.\n\nПри подозрении на проглатывание кнопочной батарейки немедленно обращайтесь за экстренной медицинской помощью, даже если симптомов нет. Не вызывайте рвоту и не ждите ухудшения состояния. Не давайте ребёнку ничего есть или пить, кроме описанного ниже исключения.\n\nЕсли батарейка может быть литиевой (или её тип неизвестен), ребёнку исполнилось 12 месяцев, предполагаемое проглатывание произошло менее 12 часов назад, ребёнок способен безопасно глотать, у него нет аллергии на мёд и мёд уже есть под рукой, по пути в больницу можно дать 10 мл (2 чайные ложки) мёда каждые 10 минут, максимум 6 доз. Не давайте мёд при рвоте или нарушении глотания. Не задерживайте выезд ради мёда: он не заменяет экстренную медицинскую помощь. При сомнениях в соблюдении условий исключения ничего не давайте внутрь и немедленно обращайтесь за помощью.',
    ),
)
CATALOG: Mapping[str, Entry] = MappingProxyType({entry.catalog_id: entry for entry in _ENTRIES})
# NO production or publisher approvals. Not an endorsement of user-provided YES.
PRODUCTION_APPROVAL_DIGESTS: Mapping[str, str] = MappingProxyType({})
_SAFE_OBJECTS = frozenset({'large_picture_card','large_soft_ball','hands'})
_WARNING_OBJECTS = frozenset({'button_battery','small_magnet'})
_REQ = frozenset({'kind','catalog_id','version','age_months','object_id','action','actor','recipient','constraints','text'})
_PROCESS_ONLY_HMAC_KEY = secrets.token_bytes(32)


def digest(entry: Entry) -> str:
    """Review must bind all entry fields, including exact UTF-8 text and age bounds."""
    return hashlib.sha256(json.dumps(asdict(entry),ensure_ascii=False,sort_keys=True,
                       separators=(',',':')).encode('utf-8')).hexdigest()


def proposal(entry: Entry, age_months: int) -> dict:
    return {'kind':entry.kind,'catalog_id':entry.catalog_id,'version':entry.version,
            'age_months':age_months,'object_id':entry.object_id,'action':entry.action,
            'actor':entry.actor,'recipient':entry.recipient,'constraints':list(entry.constraints),
            'text':entry.text}


def _admit(value: object, approvals: Mapping[str,str]) -> Decision:
    if type(value) is not dict or frozenset(value) != _REQ:
        return Decision('NO_POST','invalid_fields')
    cid=value['catalog_id']
    if type(cid) is not str or cid not in CATALOG:
        return Decision('NO_POST','unknown_catalog_id')
    entry=CATALOG[cid]
    age=value['age_months']
    if type(age) is not int or age < entry.age_min_months or (entry.age_max_months is not None and age > entry.age_max_months):
        return Decision('NO_POST','age_ineligible',cid)
    if type(value['version']) is not int or any(value[key] != getattr(entry,key)
        for key in ('kind','version','object_id','action','actor','recipient','text')) or value['constraints'] != list(entry.constraints):
        return Decision('NO_POST','entry_or_text_modified',cid)
    if entry.kind=='exercise' and entry.object_id not in _SAFE_OBJECTS:
        return Decision('NO_POST','exercise_object_unsafe',cid)
    if entry.kind=='warning' and entry.object_id not in _WARNING_OBJECTS:
        return Decision('NO_POST','warning_object_invalid',cid)
    if entry.kind not in ('warning','exercise'):
        return Decision('NO_POST','invalid_lane',cid)
    if approvals.get(cid)!=digest(entry):
        return Decision('NO_POST','trusted_digest_approval_absent',cid)
    return Decision('ALLOW','exact_frozen_template',cid,age)


def _seal(record: Rendered) -> str:
    value=asdict(record)
    value.pop('seal',None)
    msg=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return hmac.new(_PROCESS_ONLY_HMAC_KEY,msg,hashlib.sha256).hexdigest()


def _render(raw: object, approvals: Mapping[str,str]) -> Rendered:
    # Take a private one-time snapshot: no attacker-controlled dict subclass and
    # no second read of a possibly concurrent-mutated caller mapping.
    if type(raw) is not dict:
        return Rendered('NO_POST','invalid_fields')
    decision=_admit(raw.copy(),approvals)
    if decision.decision!='ALLOW':
        return Rendered('NO_POST',decision.reason)
    entry=CATALOG[decision.catalog_id]
    out=Rendered('ALLOW','exact_frozen_renderer',
        lane='safety_warning' if entry.kind=='warning' else 'exercise',
        catalog_id=entry.catalog_id,version=entry.version,age_months=decision.age_months,
        object_id=entry.object_id,action=entry.action,actor=entry.actor,
        recipient=entry.recipient,constraints=entry.constraints,text=entry.text,digest=digest(entry))
    return replace(out,seal=_seal(out))


def render(raw: object) -> Rendered:
    """PUBLIC, non-activatable: ALWAYS NO_POST until separately authorized stage."""
    return _render(raw,PRODUCTION_APPROVAL_DIGESTS)


def _validate_final(out: object, final_text: object, lane: object, approvals: Mapping[str,str]) -> Decision:
    if type(out) is not Rendered or out.decision!='ALLOW':
        return Decision('NO_POST','no_approved_render')
    if type(final_text) is not str or type(lane) is not str:
        return Decision('NO_POST','invalid_final_types')
    if not isinstance(out.seal,str) or not hmac.compare_digest(out.seal,_seal(out)):
        return Decision('NO_POST','render_seal_invalid')
    entry=CATALOG.get(out.catalog_id)
    if entry is None:
        return Decision('NO_POST','template_disappeared')
    expected=_render(proposal(entry,out.age_months),approvals)
    if expected.decision!='ALLOW' or expected!=out:
        return Decision('NO_POST','changed_immutable_metadata')
    if lane!=expected.lane or final_text.encode('utf-8')!=entry.text.encode('utf-8'):
        return Decision('NO_POST','changed_final_text_or_lane')
    return Decision('ALLOW','exact_final_text',entry.catalog_id)


def validate_final(out: object, final_text: object, lane: object) -> Decision:
    return _validate_final(out,final_text,lane,PRODUCTION_APPROVAL_DIGESTS)


def _test_fixture_only_render(raw: object, simulated_approvals: Mapping[str,str]) -> Rendered:
    """Offline only. Simulated approvals CANNOT become production approval."""
    return _render(raw,simulated_approvals)


def _test_fixture_only_validate(out: object, final_text: object, lane: object,
                               simulated_approvals: Mapping[str,str]) -> Decision:
    return _validate_final(out,final_text,lane,simulated_approvals)
