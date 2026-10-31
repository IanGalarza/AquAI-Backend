"""Nombres legibles de columnas y resultados según el paso del plan que los produjo."""

from aquai.application.answering.wording import calculation_noun, grain_label, humanize


class FieldNames:
    """Un resultado se nombra por su cálculo (máximo del pH promedio), nunca por la columna cruda que lo originó."""

    def __init__(self, steps, column, language="es"):
        self._steps, self._column, self._language = steps, column, language

    def origin(self, step):
        """Ordenar o elegir el último no cambia las columnas: su nombre viene del paso que las calculó."""
        while step is not None and step.operation in {"sort", "latest"} and step.depends_on:
            step = self._steps.get(step.depends_on[0])
        return step

    def column(self, dataset_id, sql_name):
        return self._column(dataset_id, sql_name) or humanize(sql_name)

    def operand(self, ref):
        if ref.kind == "column":
            return self.column(ref.dataset_id, ref.column)
        return self.result(ref.step_id, ref.field)

    def result(self, step_id, name, field=None):
        step = self.origin(self._steps.get(step_id))
        if step is not None:
            calculation = next((item for item in step.calculations if item.output_name == name), None)
            if calculation is not None:
                return self.calculation(calculation)[0]
            grouping = next((item for item in step.group_by if item.output_name == name), None)
            if grouping is not None:
                if grouping.time_grain != "raw":
                    return grain_label(grouping.time_grain, self._language)
                if grouping.field is not None:
                    return self.operand(grouping.field)
            field = field or next((item for item in step.output_fields if item.name == name), None)
        labels = {self.column(ref.dataset_id, ref.column) for ref in field.sources} if field is not None else set()
        return labels.pop() if len(labels) == 1 else humanize(name)

    def calculation(self, calculation):
        """Sustantivo y género del cálculo, con los nombres de sus operandos."""
        counted = calculation.function == "count" and not calculation.inputs
        variables = [] if counted else [self.operand(ref) for ref in calculation.inputs]
        return calculation_noun(calculation.function, variables, self._language,
                                quantile=calculation.quantile, ddof=calculation.ddof)
