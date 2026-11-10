from aquai.application.contracts.planning import MAX_ROW_LIMIT
from aquai.application.matching import contains_any_phrase, mentioned_count
from aquai.application.Rules.vocabulary import FULL_ORDER_CUES

_MEASURED_TYPES = {"NUMERIC", "BIGINT"}


#
# Devuelve la cantidad de filas pedida en las menciones de límite, si es única y válida.
#
def requested_limit(mentions) -> int | None:
    counts = [mentioned_count(item.text) for item in _limits(mentions)]
    if not counts or None in counts or len(set(counts)) != 1:
        return None

    return counts[0] if counts[0] <= MAX_ROW_LIMIT else None


#
# Devuelve las menciones de límite, sin las que sólo piden un orden («ordenados de mayor a menor»).
#
def _limits(mentions) -> list:
    return [
        item
        for item in mentions
        if item.kind == "limit"
        and (mentioned_count(item.text) is not None or not contains_any_phrase(item.text, FULL_ORDER_CUES))
    ]


#
# Devuelve las menciones para el Planner, descartando los límites sin número.
#
def countable_mentions(mentions) -> list:
    return [
        item for item in mentions if item.kind != "limit" or mentioned_count(item.text) is not None
    ]


#
# Devuelve los límites del resultado citado por clase de selección, salvo que se pida otro.
#
def conserved_limits(mentions, analysis) -> dict[tuple, int]:
    if analysis is None or _limits(mentions):
        return {}

    steps = {item["id"]: item for item in analysis.operations}
    outputs = [item.get("operation_id") for item in analysis.operation_references]
    limits = {}

    for chain, kind in _published_selections(steps, outputs):
        values = [steps[key]["limit"] for key in chain if steps[key].get("limit")]
        if values:
            limits.setdefault(kind, min(values))

    return limits


#
# Devuelve el límite que corresponde a cada paso que elige filas publicadas.
#
def result_limits(steps, outputs, requested, conserved) -> dict[str, int]:
    targets = {}

    for chain, kind in _published_selections(steps, outputs):
        if requested is not None:
            targets.update(dict.fromkeys(chain, requested))
        elif kind in conserved and not any(steps[key].get("limit") for key in chain):
            targets[chain[0]] = conserved[kind]

    return {key: value for key, value in targets.items() if steps[key].get("limit") != value}


#
# Recorre las selecciones de filas de las salidas publicadas y su clase.
#
def _published_selections(steps, outputs):
    for key in dict.fromkeys(outputs):
        if key not in steps:
            continue

        chain, kind = _selection(_published_rows(key, steps), steps)
        if chain:
            yield chain, kind


#
# Devuelve el paso cuyas filas publica una salida; un gráfico publica las de su entrada.
#
def _published_rows(key, steps):
    step = steps[key]
    if step["operation"] == "visualize" and step["depends_on"]:
        return step["depends_on"][0]

    return key


#
# Devuelve los pasos que eligen las filas de un resultado y su clase de selección.
#
def _selection(key, steps):
    chain, origin = _sort_chain(key, steps)
    if not chain and origin in steps and steps[origin]["operation"] == "read":
        chain = [origin]

    ancestry = _ancestry(origin, steps)
    shared = _used_by_other_branch(chain, steps)
    if not chain or shared or _follows_latest(ancestry, steps):
        return [], None

    grouped = any(steps[item]["operation"] in {"aggregate", "calculate"} for item in ancestry)
    selector = steps[chain[0]]

    return chain, (selector["operation"], grouped, _ordered_by_measure(selector))


#
# Devuelve la cadena de sort contiguos de un paso y el paso donde termina.
#
def _sort_chain(key, steps):
    chain = []

    while _continues_sort_chain(key, chain, steps):
        chain.append(key)
        key = steps[key]["depends_on"][0]

    return chain, key


#
# Indica si el paso continúa la cadena de sort.
#
def _continues_sort_chain(key, chain, steps) -> bool:
    return bool(
        key in steps
        and key not in chain
        and steps[key]["operation"] == "sort"
        and steps[key]["depends_on"]
    )


#
# Indica si otra rama usa las filas de la cadena, de modo que limitarlas cambiaría ese resultado.
#
def _used_by_other_branch(chain, steps) -> bool:
    for item in chain:
        for step in steps.values():
            if item not in step["depends_on"]:
                continue
            if step["operation"] != "visualize" and step["id"] not in chain:
                return True

    return False


#
# Indica si algún paso del linaje es un latest.
#
def _follows_latest(ancestry, steps):
    return any(steps[item]["operation"] == "latest" for item in ancestry)


#
# Indica si el paso ordena por una variable numérica medida.
#
def _ordered_by_measure(step):
    if not step.get("order_by"):
        return False

    ref = step["order_by"][0]["field"]
    name = ref.get("field") or ref.get("column")

    return any(
        field["name"] == name and field["sql_type"] in _MEASURED_TYPES
        for field in step.get("output_fields") or []
    )


#
# Devuelve el paso indicado y todos los pasos de los que depende.
#
def _ancestry(key, steps):
    found, pending = set(), [key]

    while pending:
        item = pending.pop()
        if item in steps and item not in found:
            found.add(item)
            pending.extend(steps[item]["depends_on"])

    return found
