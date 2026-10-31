from aquai.application.contracts.planning import MAX_ROW_LIMIT
from aquai.application.matching import mentioned_count

_MEASURED_TYPES = {"NUMERIC", "BIGINT"}


def requested_limit(mentions) -> int | None:
    """Cantidad pedida en la consulta («los primeros 5 registros»).

    Varias cantidades distintas, o una mención sin número («todos los registros»), no fijan ninguna."""
    counts = [mentioned_count(item.text) for item in mentions if item.kind == "limit"]
    if not counts or None in counts or len(set(counts)) != 1:
        return None

    return counts[0] if counts[0] <= MAX_ROW_LIMIT else None


def countable_mentions(mentions) -> list:
    """Menciones para el Planner: una cantidad sin número («todos», un superlativo mal clasificado) no es un limit."""
    return [
        item for item in mentions if item.kind != "limit" or mentioned_count(item.text) is not None
    ]


def conserved_limits(mentions, analysis) -> dict[tuple, int]:
    """Límites del resultado citado por clase de selección; mencionar una cantidad ahora los reemplaza."""
    if analysis is None or any(item.kind == "limit" for item in mentions):
        return {}

    steps = {item["id"]: item for item in analysis.operations}
    outputs = [item.get("operation_id") for item in analysis.operation_references]
    limits = {}

    for chain, kind in _published_selections(steps, outputs):
        values = [steps[key]["limit"] for key in chain if steps[key].get("limit")]
        if values:
            limits.setdefault(kind, min(values))

    return limits


def result_limits(steps, outputs, requested, conserved) -> dict[str, int]:
    """Límite de cada paso que elige filas publicadas.

    La cantidad pedida acota toda selección publicada; sin pedido, una selección sin límite conserva
    el del resultado citado de su misma clase. steps son pasos validados serializados, por id."""
    targets = {}

    for chain, kind in _published_selections(steps, outputs):
        if requested is not None:
            targets.update(dict.fromkeys(chain, requested))
        elif kind in conserved and not any(steps[key].get("limit") for key in chain):
            targets[chain[0]] = conserved[kind]

    return {key: value for key, value in targets.items() if steps[key].get("limit") != value}


def _published_selections(steps, outputs):
    """Selecciones de las filas publicadas; un gráfico publica las filas de su entrada."""
    for key in dict.fromkeys(outputs):
        if key not in steps:
            continue  # La validación rechaza salidas inexistentes.

        chain, kind = _selection(_published_rows(key, steps), steps)
        if chain:
            yield chain, kind


def _published_rows(key, steps):
    step = steps[key]
    if step["operation"] == "visualize" and step["depends_on"]:
        return step["depends_on"][0]

    return key


def _selection(key, steps):
    """Pasos que eligen las filas de key: los sort contiguos que las ordenan o el read publicado.

    La clase distingue registros sin orden, registros o grupos ordenados por un valor medido y ordenados por
    otra cosa (una fecha, un nombre): un límite sólo se conserva dentro de la misma clase, así el mayor valor
    no recorta una evolución temporal. Tras latest no hay selección, porque un límite cortaría sus empates."""
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


def _sort_chain(key, steps):
    chain = []

    while _continues_sort_chain(key, chain, steps):
        chain.append(key)
        key = steps[key]["depends_on"][0]

    return chain, key


def _continues_sort_chain(key, chain, steps) -> bool:
    return bool(
        key in steps
        and key not in chain
        and steps[key]["operation"] == "sort"
        and steps[key]["depends_on"]
    )


def _used_by_other_branch(chain, steps) -> bool:
    """Si otra rama usa estas filas (un aggregate, otro sort), limitarlas cambiaría también ese resultado."""
    for item in chain:
        for step in steps.values():
            if item not in step["depends_on"]:
                continue
            if step["operation"] != "visualize" and step["id"] not in chain:
                return True

    return False


def _follows_latest(ancestry, steps):
    return any(steps[item]["operation"] == "latest" for item in ancestry)


def _ordered_by_measure(step):
    if not step.get("order_by"):
        return False

    ref = step["order_by"][0]["field"]
    name = ref.get("field") or ref.get("column")

    return any(
        field["name"] == name and field["sql_type"] in _MEASURED_TYPES
        for field in step.get("output_fields") or []
    )


def _ancestry(key, steps):
    found, pending = set(), [key]

    while pending:
        item = pending.pop()
        if item in steps and item not in found:
            found.add(item)
            pending.extend(steps[item]["depends_on"])

    return found
