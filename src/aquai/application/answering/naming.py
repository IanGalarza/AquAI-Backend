from aquai.application.answering.wording import calculation_noun, grain_label, humanize


class FieldNames:
    #
    # Guarda los pasos del plan, el resolvedor de columnas y el idioma.
    #
    def __init__(self, steps, column, language="es"):
        self._steps = steps
        self._column = column
        self._language = language

    #
    # Devuelve el paso que calculó las columnas, saltando ordenamientos y últimos.
    #
    def origin(self, step):
        while step is not None and step.operation in {"sort", "latest"} and step.depends_on:
            step = self._steps.get(step.depends_on[0])

        return step

    #
    # Devuelve el nombre de una columna, o uno legible si no hay etiqueta.
    #
    def column(self, dataset_id, sql_name):
        return self._column(dataset_id, sql_name) or humanize(sql_name)

    #
    # Devuelve el nombre de un operando, sea columna o resultado de un paso.
    #
    def operand(self, ref):
        if ref.kind == "column":
            return self.column(ref.dataset_id, ref.column)

        return self.result(ref.step_id, ref.field)

    #
    # Devuelve el nombre de un campo de resultado según el cálculo que lo originó.
    #
    def result(self, step_id, name, field=None):
        step = self.origin(self._steps.get(step_id))
        if step is not None:
            planned = self._planned_name(step, name)
            if planned is not None:
                return planned
            field = field or next((item for item in step.output_fields if item.name == name), None)

        return self._source_name(field, name)

    #
    # Devuelve el nombre de un campo según el cálculo o la agrupación planificados.
    #
    def _planned_name(self, step, name):
        calculation = next((item for item in step.calculations if item.output_name == name), None)
        if calculation is not None:
            return self.calculation(calculation)[0]

        grouping = next((item for item in step.group_by if item.output_name == name), None)
        if grouping is None:
            return None
        if grouping.time_grain != "raw":
            return grain_label(grouping.time_grain, self._language)
        if grouping.field is not None:
            return self.operand(grouping.field)

        return None

    #
    # Devuelve el nombre de un campo según sus columnas de origen.
    #
    def _source_name(self, field, name):
        if field is None:
            return humanize(name)

        labels = {self.column(ref.dataset_id, ref.column) for ref in field.sources}

        return labels.pop() if len(labels) == 1 else humanize(name)

    #
    # Devuelve el nombre de un cálculo y si es femenino.
    #
    def calculation(self, calculation):
        counted = calculation.function == "count" and not calculation.inputs
        variables = [] if counted else [self.operand(ref) for ref in calculation.inputs]

        return calculation_noun(
            calculation.function,
            variables,
            self._language,
            quantile=calculation.quantile,
            ddof=calculation.ddof,
        )
