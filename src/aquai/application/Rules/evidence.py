import re

CONDITIONAL_CUES = re.compile(
    r"\b(?:si|de\s+ser|en\s+caso\s+de|suponiendo|podr[ií]a|podr[ií]an|quedar[ií]an?|ser[ií]an?|estar[ií]an?|"
    r"equivaldr[ií]an?|hip[oó]tesis|hipot[eé]tic[oa]s?|if|assuming|would|could|might|hypothes[ie]s|"
    r"hypothetical(?:ly)?)\b",
    re.IGNORECASE,
)
HEDGE_CUES = re.compile(
    r"\b(?:podr[ií]a|podr[ií]an|puede|pueden|posible|posiblemente|quiz[aá]s?|tal\s+vez|suele|suelen|"
    r"se\s+asocia|se\s+relaciona|se\s+vincula|sugiere|sugieren|could|might|may|possibly|perhaps|"
    r"associated|related|suggests?|tends?)\b",
    re.IGNORECASE,
)
VERDICT_LIMITS = re.compile(
    r"\b(?:si\s+(?:el|la|esta|esa)\b|no\s+permite|no\s+alcanza|no\s+puede|no\s+se\s+puede|no\s+es\s+posible|"
    r"no\s+hay\s+evidencia|sin\s+que|whether|cannot|can\s+not|does\s+not\s+allow|not\s+enough)\b",
    re.IGNORECASE,
)
UNSUPPORTED_VERDICTS = re.compile(
    r"\b(?:el\s+agua|esta\s+agua|esa\s+agua|las\s+aguas|dicha\s+agua|the\s+water|this\s+water)\b"
    r"[^.;\n]{0,60}?\b(?:es|est[aá]|son|est[aá]n|resulta|resultan|is|are)\s+(?:\w+\s+){0,2}?"
    r"(?:aptas?|seguras?|inocuas?|potables?|t[oó]xicas?|peligrosas?|contaminadas?|insalubres?|"
    r"safe|unsafe|potable|toxic|hazardous|contaminated|fit\s+for)\b"
    r"|\b(?:no\s+(?:hay|existe|representa|presenta)\s+(?:ning[uú]n\s+)?riesgo|"
    r"sin\s+riesgo\s+(?:para|sanitario)|poses?\s+no\s+risk|no\s+health\s+risk)\b",
    re.IGNORECASE,
)
# Lo que sigue a «provienen de» cuando nombra de dónde salen los datos y no una causa («provienen de archivos
# distintos», «come from two sheets»).
_DATA_ORIGIN = (
    r"(?!\s+(?:\w+\s+){0,2}?(?:archivos?|fuentes?|hojas?|planillas?|tablas?|datasets?|bases?\s+de\s+datos|files?|"
    r"sources?|sheets?|tables?)\b)"
)
CAUSAL_CLAIMS = re.compile(
    r"\b(?:se\s+deb(?:e|en|i[oó]|ieron)\s+a|debid[oa]s?\s+a|a\s+causa\s+de|causad[oa]s?|caus[oó]|provocad[oa]s?|"
    r"provoc[oó]|origin[oó]|producto\s+de|consecuencia\s+de|por\s+culpa\s+de|gracias\s+a|explicad[oa]s?\s+por|"
    rf"originad[oa]s?\s+por|generad[oa]s?\s+por|producid[oa]s?\s+por|obedecen?\s+a|responden?\s+a|provienen?\s+de{_DATA_ORIGIN}|"
    rf"derivan?\s+de|resultan?\s+de|se\s+origina(?:n|ron)?|results?\s+from|stems?\s+from|comes?\s+from{_DATA_ORIGIN}|"
    r"originat(?:es?|ed)\s+from|generated\s+by|produced\s+by|a\s+consequence\s+of|"
    r"se\s+explica\s+por|atribuid[oa]s?\s+a|atribuible\s+a|demuestra\s+que|prueba\s+que|is\s+due\s+to|was\s+due\s+to|"
    r"due\s+to|because\s+of|caused(?:\s+by)?|as\s+a\s+result\s+of|owing\s+to|explained\s+by|attributable\s+to|"
    r"driven\s+by|proves\s+that|demonstrates\s+that)\b",
    re.IGNORECASE,
)
SENTENCE_END = re.compile(r"(?<=[.!?;])\s+|\n+")
DOCUMENT_ANAPHORA = re.compile(
    r"\b(?:es[eao]s?|dich[oa]s?|mism[oa]s?|anterior(?:es)?|that|those|this|these|same|previous)\s+"
    r"(?:\w+\s+)?(?:documentos?|papers?|art[ií]culos?|estudios?|gu[ií]as?|fuentes?|autor(?:es)?|"
    r"tesis|informes?|documents?|articles?|stud(?:y|ies)|guidelines?|sources?|authors?|reports?)\b",
    re.IGNORECASE,
)
DOCUMENT_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
DOCUMENT_AUTHOR_NAMES = (
    re.compile(r"([^\W\d_]+)(?=\s+et\s+al)", re.IGNORECASE),
    re.compile(r"^([^\W\d_]+)(?=\s*(?:19|20)\d{2})"),
    re.compile(r"^([^\W\d_]+)\s+y\s+[^\W\d_]+,"),
    re.compile(r"^[^\W\d_]+\s+y\s+([^\W\d_]+),"),
    re.compile(r"\(([A-Z]{2,})\)"),
    re.compile(r"^(?:[A-Z]{3,}\s+)*([A-Z]{3,})(?=[\s~_(])"),
)
MARKUP = re.compile(
    r"\*\*|__|^\s*#{1,6}\s|`|^\s*\|.+\|\s*$|https?://|www\.|<[a-zA-Z/][^>]*>|\S+@\S+\.\S+",
    re.MULTILINE,
)
FOREIGN_SCRIPT = re.compile(r"[\u0400-\u04ff\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")
INLINE_REFERENCE = re.compile(r"\[\s*[a-z]?\d{1,2}\s*\]|(?<![\w-])[dsec]\d{1,2}(?![\w-])")
UNSUPPORTED_STATISTICS = re.compile(
    r"\b(?:significativ[oa]s?|estad[ií]sticamente|tendencia|tendencias|correlaci[oó]n|"
    r"statistically|significant|significantly|trend|correlation)\b",
    re.IGNORECASE,
)
REFERENCE_LINE_CUES = re.compile(
    r"\b(?:(?:valor|nivel)(?:es)?\s+(?:gu[ií]a|de\s+referencia|l[ií]mite|de\s+alerta)|l[ií]nea\s+de\s+referencia|"
    r"umbral(?:es)?|l[ií]mite(?:s)?\s+(?:normativo|permitido|recomendado|de\s+referencia)s?|norma(?:s|tiva)?|"
    r"guideline|guidance\s+levels?|alert\s+levels?|reference\s+(?:line|value)|threshold)\b",
    re.IGNORECASE,
)
WATER_USES = {
    "drinking": (
        "agua potable",
        "agua de bebida",
        "agua de consumo",
        "agua para consumo",
        "consumo humano",
        "agua de red",
        "drinking water",
        "drinking-water",
        "potable water",
        "tap water",
    ),
    "recreational": (
        "uso recreativo",
        "aguas recreativas",
        "agua recreativa",
        "recreacional",
        "recreacion",
        "balneario",
        "bañistas",
        "recreational",
        "bathing",
        "swimming",
    ),
    "irrigation": ("riego", "irrigation"),
    "aquatic_life": ("vida acuatica", "biota acuatica", "aquatic life"),
}
PRINTED_FIGURE_TEXT = re.compile(r"^Texto impreso en", re.IGNORECASE)
SEARCH_STOPWORDS = frozenset(
    """a al algo alguna algunas alguno algunos ante antes aqui como con contra cual cuales cuando
    cuanto de del desde donde durante e el ella ellas ello ellos en entre era eran es esa esas ese eso
    esos esta estaba estado estan estar estas este esto estos fue fueron ha hay la las le les lo los mas
    me mi muy no nos o otra otras otro otros para pero por porque puede pueden que quien se segun ser si
    sin sobre son su sus tambien tan te tener tiene tienen tu un una uno unos y ya the of and in to was
    were with for that this from is are be by as at an or on it its which these those have has had not
    can may might will would their there been between than into such also but we our they them more
    most other some any what how why when where who whom do does did about""".split()
)
FULL_DATE = re.compile(r"\b\d{1,2}/\d{1,2}/(\d{4})\b")
ABBREVIATION_END = re.compile(
    r"(?:\b(?:et\s+al|al|etc|figs?|tabs?|refs?|pp?|spp?|var|cf|ca|aprox|dra?|sra?|vs|nro|ej)\.|(?<![-\w])[A-Z]\.)$",
    re.IGNORECASE,
)
OBSERVED_CLASSIFICATION = re.compile(
    r"\b(?:(?:rango|valor(?:es)?|nivel(?:es)?)\b[^.;\n]{0,60}\b(?:normal(?:es)?|segur[oa]s?|aceptables?|"
    r"recomendad[oa]s?|adecuad[oa]s?|saludables?)|"
    r"(?:considerad[oa]s?|es|son|resulta|resultan)\s+(?:segur[oa]s?|normal(?:es)?|aceptables?|"
    r"adecuad[oa]s?|saludables?|[oó]ptim[oa]s?)|riesgos?\s+(?:inmediatos?|para\s+la\s+salud)|"
    r"within\s+the\s+(?:safe|normal|acceptable|recommended)|"
    r"(?:is|are|considered)\s+(?:safe|normal|acceptable|adequate|optimal)|health\s+risks?)\b",
    re.IGNORECASE,
)
OBSERVED_COMPARISON = re.compile(
    r"\b(?:dentro\s+(?:del?\s+)?(?:un\s+|el\s+|los\s+)?(?:rango|intervalo|l[ií]mites?|par[aá]metros?|valores?)|"
    r"super(?:a|an|ar|aron|ó|ior(?:es)?)\b|exced(?:e|en|er|ieron|ió)\b|por\s+(?:debajo|encima)\s+de|"
    r"inferior(?:es)?\b|exceed(?:s|ed)?\b|above\s+the|below\s+the|higher\s+than|lower\s+than)\b",
    re.IGNORECASE,
)
MEAN_WITH_DEVIATION = re.compile(r"(?P<mean>\d+(?:[.,]\d+)?)\s*±\s*\d")
DEICTIC_PLACE = re.compile(
    r"\b(?:este|ese|dicho|mismo)\s+(?:sitio|lugar|punto|estaci[oó]n)\b|"
    r"\b(?:esta|esa|dicha|misma)\s+(?:estaci[oó]n|zona|localidad)\b|"
    r"\b(?:this|that|the\s+same)\s+(?:site|location|place|station)\b",
    re.IGNORECASE,
)
REFERENTIAL_QUESTION = re.compile(
    r"\b(?:eso|esto|ese|esa|este|esta|aquello|lo\s+anterior|ese\s+resultado|that|this|it)\b", re.IGNORECASE
)
REFERENCE_REQUEST_CLAUSE = re.compile(
    r"(?:\s*[,;]\s*|\s+(?:y|e|con|and|with)\s+)"
    r"(?:(?:agreg|sum|marc|inclu|mostr|dibuj|traz|pon|add|includ|show|mark|draw)\w*\s+)?"
    r"(?:(?:el|la|los|las|un|una|su|sus|the|a|an)\s+)?"
    r"(?:(?:l[ií]nea|valor(?:es)?)\s+(?:gu[ií]a|de\s+referencia|l[ií]mite)|umbral(?:es)?|"
    r"l[ií]mite(?:s)?\s+(?:normativo|permitido|recomendado|de\s+referencia)s?|normas?\b|"
    r"guideline(?:\s+value)?s?|reference\s+(?:line|value)s?|thresholds?)"
    r"(?![^.;?!]*\d)"
    r"(?:(?!\s(?:en|durante|desde|hasta|entre|por|in|during|from|until|since)\b)[^.;?!,])*",
    re.IGNORECASE,
)
LEADING_CONNECTOR = re.compile(
    r"^(?:por\s+otro\s+lado|sin\s+embargo|adem[aá]s|asimismo|por\s+lo\s+tanto|por\s+ello|en\s+cambio|"
    r"no\s+obstante|tambi[eé]n|however|moreover|in\s+addition|furthermore|therefore|also),?\s+",
    re.IGNORECASE,
)
CITATION_QUESTION = re.compile(
    r"\b(?:en\s+)?qu[eé]\s+p[aá]gina\b|\bde\s+d[oó]nde\s+(?:sacaste|lo\s+sacaste|sale\s+eso|salen\s+esos|sali[oó]\s+eso)\b|"
    r"\bqu[eé]\s+(?:documentos?|fuentes?|papers?|art[ií]culos?)\s+(?:usaste|citaste|consultaste)\b|"
    r"\bcu[aá]l(?:es)?\s+(?:es|son)\s+(?:la|las)\s+fuentes?\s+(?:de\s+eso|que\s+usaste|de\s+esa\s+respuesta)\b|"
    r"\bwhich\s+pages?\b|\bwhat\s+pages?\b|\bwhere\s+does\s+it\s+say\b|\bwhat\s+sources?\s+did\s+you\b",
    re.IGNORECASE,
)
REPEAT_REQUEST = re.compile(
    r"^\W*(?:rep[ií]te(?:lo|me)?|repet[ií](?:lo|me)?|repet[ií]melo|volv[eé]\s+a\s+mostrar\w*|mostr[aá]\w*\s+de\s+nuevo|"
    r"repeat|show\s+(?:it|that)\s+again)\b",
    re.IGNORECASE,
)
OBSERVED_VERDICTS = re.compile(
    r"\b(?:apt[oa]s?|potables?|segur[oa]s?|inocu[oa]s?|insalubres?|t[oó]xic[oa]s?|peligros[oa]s?|"
    r"contaminad[oa]s?|cumple[n]?\s+con|incumple[n]?|se\s+puede\s+(?:beber|bañar|tomar|consumir)|"
    r"no\s+representa[n]?\s+(?:peligro|riesgo)|sin\s+problemas|buena\s+calidad|mala\s+calidad|"
    r"(?:baj[oa]|alt[oa]|menor|mayor)\s+(?:probabilidad|riesgo)|efectos?\s+(?:adversos?|nocivos?)|"
    r"anomal[ií]as?|an[oó]mal[oa]s?|at[ií]pic[oa]s?|lo\s+(?:normal|esperad[oa]|esperable|habitual)|"
    r"anomal(?:y|ies|ous)|atypical|(?:than\s+)?expected|unusual|"
    r"safe|unsafe|potable|drinkable|toxic|hazardous|contaminated|fit\s+for|complies|good\s+quality|poor\s+quality|"
    r"(?:low|high)(?:er)?\s+(?:probability|risk)|adverse\s+effects?)\b",
    re.IGNORECASE,
)
RECOMMENDATIONS = re.compile(
    r"\b(?:se\s+(?:recomiend\w*|sugier\w*|aconsej\w*|deber[ií]an?|debe|deben)|recomend(?:amos|ar[ií]a|able)|"
    r"conviene|convendr[ií]a|hay\s+que|habr[ií]a\s+que|es\s+(?:necesario|recomendable|aconsejable|conveniente)|"
    r"deber[ií]an?\s+(?:informar|cerrar|prohibir|evitar|restringir|tomar|realizar|monitorear|controlar)|"
    r"should|must|ought\s+to|(?:is|are)\s+recommended|we\s+recommend|it\s+is\s+(?:necessary|advisable))\b",
    re.IGNORECASE,
)
OBSERVED_REFERENCE = re.compile(
    r"\b(?:este\s+(?:resultado|valor|promedio)|estos\s+(?:resultados|valores)|esta\s+agua|estas\s+aguas|"
    r"(?:el\s+)?valor\s+observado|lo\s+observado|lo\s+(?:ya\s+)?calculado|los\s+datos\s+analizados|en\s+los\s+datos|"
    r"los\s+datos\s+(?:muestran|indican|sugieren|revelan|reflejan)|"
    r"(?:los|el)\s+(?:valor(?:es)?|resultados?|datos?)\s+(?:observad|registrad|medid|calculad|obtenid)\w*"
    r"(?!\s+(?:en|por|durante|entre|para|de)\b)|this\s+(?:result|value|average|water)|"
    r"the\s+observed\s+values?|the\s+analy[sz]ed\s+data)\b",
    re.IGNORECASE,
)
RELATIVE_QUANTITIES = re.compile(
    r"\b(?:el\s+doble|el\s+triple|la\s+mitad|un\s+tercio|(?:dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|cien)\s+veces|"
    r"\w+\s+veces\s+(?:m[aá]s|menos|mayor|menor)|twice|double|triple|half\s+of|\w+\s+times\s+(?:more|less|higher|lower))\b",
    re.IGNORECASE,
)
# Sentido de una comparación con una referencia. Los verbos van por familia («superaron», «sobrepasan», «rebasó»);
# «pasar» sólo compara seguido de «de» o de un artículo («pasaron los niveles guía», «pasa de 100»), no en «qué pasó
# con»; «no llegar» o «no alcanzar» un nivel es quedar por debajo.
COMPARISON_DIRECTIONS = {
    "above": re.compile(
        r"\b(?:por\s+encima|super(?:a|an|ar|aron|ó|ior(?:es)?)\b|exced\w*|sobrepas\w*|rebas\w*|"
        r"pas(?:a|an|ar|ó|aron|e|en|ara|aran)\s+(?:de|del|los|las|el|la|lo)\b|mayor(?:es)?\b|above|exceed\w*|"
        r"higher)\b",
        re.IGNORECASE,
    ),
    "below": re.compile(
        r"\b(?:por\s+debajo|inferior(?:es)?\b|menor(?:es)?\b|no\s+(?:lleg|alcanz)\w*|below|lower)\b", re.IGNORECASE
    ),
    "within": re.compile(r"\b(?:dentro\s+del?|within)\b", re.IGNORECASE),
}
# Comparaciones que incluyen el límite: «alcanzaron o superaron», «igual o mayor», «al menos», «or more».
INCLUSIVE_BOUND = re.compile(
    r"\b(?:(?:alcanz|iguala|lleg)\w*\s+o\s+(?:super|exced|sobrepas|rebas|pas)\w*|(?:super|exced|sobrepas|rebas|pas)"
    r"\w*\s+o\s+(?:alcanz|iguala|lleg)\w*|(?:mayor|menor|superior|inferior)(?:es)?\s+o\s+igual\w*|igual\w*\s+o\s+"
    r"(?:mayor|menor|superior|inferior)\w*|al\s+menos|como\s+m[ií]nimo|como\s+m[aá]ximo|o\s+m[aá]s|o\s+menos|"
    r"at\s+least|at\s+most|or\s+(?:more|less|above|below)|reach\w*\s+or\s+exceed\w*|"
    r"equal\s+(?:to\s+)?or\s+(?:greater|higher|above|less|lower|below))\b|[≥≤]",
    re.IGNORECASE,
)
RESULT_WORDS = frozenset(
    """a al alcanzo alcanzaron ano anos aparece aparecen archivo calculado calculada calculados calculadas cada
    como con correspondiente correspondientes dato datos de del dicho durante e el en entre es esa ese esta este
    estos estas fecha fechas fue fueron grafico grafica graficaron graficos hoja la las lo los lugar lugares medido
    medida medidos medidas mes meses mostrando muestra muestran mostro numero o observado observada observados
    observadas obtenido obtenida obtenidos para por presenta presentan que registrado registrada registrados
    registradas registro registros resultado resultados respectivamente se segun ser sitio sitios son su sus tabla total
    un una unos unas valor valores y ya analizado analizados analizada analizadas calcularon calculo calcular
    obteniendo obtuvo obtuvieron disponible disponibles periodo periodos cual cuales donde puede pueden consultar
    consultarse presentando representa representan ver verse excluyendo excluidos excluidas hubo hay
    registraron identificaron encontraron superior superiores inferior inferiores otro otra lado parte mientras
    a an and are as at by chart data during file for from in is it its measured obtained observed of on or per
    place recorded records registered result results sheet site sites table that the this to value values was
    were with year years""".split()
)
#
# Afirmaciones de que no existen datos («No hay información disponible sobre…»): que el análisis no haya podido
# obtener algo no prueba que no exista, así que un párrafo de resultado sólo puede decirlo si un hecho lo dice.
#
ABSENCE_CLAIM = re.compile(
    r"\b(?:no\s+(?:hay|hab[ií]a|existen?|figuran?|se\s+(?:registr|encontr|dispon|report|midi|cuenta)\w*)\s+"
    r"(?:\w+\s+){0,3}?(?:informaci[oó]n|datos?|registros?|mediciones|valores|recuentos?)|"
    r"(?:there\s+(?:is|are|was|were)\s+no|no)\s+(?:\w+\s+){0,2}?(?:data|records?|information|measurements?)\s+"
    r"(?:is|are|was|were|available|exists?))\b",
    re.IGNORECASE,
)
EMPTY_RESULT = re.compile(r"\b(?:no\s+(?:encontr|hay|tiene|se\s+pudo)\w*|sin\s+(?:dato|valor)\w*|no\s+records?)\b", re.IGNORECASE)
#
# Valoraciones de lo observado que un párrafo de resultado sólo puede hacer si sus hechos las dicen: intensidad,
# nivel, cambio, estabilidad, tipicidad y relación. Se comparan palabras normalizadas (sin tildes); la intensidad,
# el nivel y el sentido del cambio se distinguen por grado, porque «débil» no autoriza «fuerte».
#
ASSESSMENT_FAMILIES = {
    "fuerte": re.compile(
        r"fuertes?|fuertemente|marcad[oa]s?|marcadamente|pronunciad[oa]s?|considerables?|considerablemente|"
        r"sustancial(?:es|mente)?|notables?|notablemente|strong(?:er|est|ly)?|marked(?:ly)?|pronounced|"
        r"considerabl[ey]|substantial(?:ly)?|notabl[ey]"
    ),
    "moderado": re.compile(r"moderad[oa]s?|moderadamente|moderate(?:ly)?"),
    "debil": re.compile(
        r"debil(?:es|mente)?|leves?|levemente|liger[oa]s?|ligeramente|escas[oa]s?|escasamente|weak(?:er|est|ly)?|"
        r"slight(?:ly)?|mild(?:ly)?"
    ),
    "nulo": re.compile(r"nul[oa]s?|inexistentes?|despreciables?|negligibl[ey]|nonexistent"),
    "alto": re.compile(
        r"alt[oa]s?|altisim[oa]s?|elevad[oa]s?|maxim[oa]s?|high(?:er|est)?|elevated|maximum|peak"
    ),
    "bajo": re.compile(r"baj[oa]s?|bajisim[oa]s?|reducid[oa]s?|minim[oa]s?|low(?:er|est)?|minimum"),
    "aumento": re.compile(
        r"aument\w*|crec\w*|ascend\w*|ascenso|sub(?:e|en|io|ieron|ir)|increment\w*|increas\w*|ris(?:e|es|ing)|"
        r"rose|grow\w*|grew|upward"
    ),
    "descenso": re.compile(
        r"disminu\w*|decrec\w*|descend\w*|descenso|reduj\w*|reducc\w*|reduce|reducen|reducir|caida|caidas|cayo|"
        r"cayeron|decreas\w*|declin\w*|fell|fall(?:s|ing)?|downward|drop\w*"
    ),
    "estable": re.compile(
        r"estab(?:le|les|ilidad|ilizo|ilizaron)|constantes?|invariab\w*|stable|stability|constant|unchanged|steady"
    ),
    "tipico": re.compile(
        r"habitual(?:es|mente)?|tipic[oa]s?|tipicamente|frecuentes?|frecuentemente|normal(?:es|mente)?|"
        r"esperables?|esperad[oa]s?|excepcional(?:es|mente)?|inusual(?:es|mente)?|extrem[oa]s?|atipic[oa]s?|"
        r"anomal[oa]s?|outliers?|typical(?:ly)?|usual(?:ly)?|unusual(?:ly)?|frequent(?:ly)?|normal(?:ly)?|"
        r"expected|exceptional(?:ly)?|extreme(?:ly|s)?|atypical|anomalous"
    ),
    "relacion": re.compile(
        r"relacion\w*|asoci\w*|vincul\w*|depend\w*|correlacion\w*|correlaciona\w*|influ\w*|efectos?|impact\w*|"
        r"related|relationships?|relation|relations|associat\w*|linked|correlat\w*|effects?"
    ),
}
#
# Comienzo de una oración que habla de un estudio, un resultado o una observación sin nombrar de quién: sin la
# oración que presentaba la fuente se leería como un análisis propio («El análisis incluyó…», «Se observó…»).
#
UNNAMED_STUDY_START = re.compile(
    r"^(?:(?:el|la|los|las|este|esta|estos|estas|dich[oa]s?|the|this|these|our)\s+"
    r"(?:an[aá]lisis|estudios?|trabajos?|investigaci[oó]n(?:es)?|muestreos?|monitoreos?|resultados?|datos|valores|"
    r"mediciones|concentraciones|autor(?:es|as?)?|study|studies|analysis|work|results?|data|values|measurements|"
    r"concentrations|authors?|survey|sampling)\b|se\s+(?:observ|registr|midi|mid|encontr|detect|hall|report|analiz|"
    r"estudi|evalu|determin|compar|realiz|aplic|utiliz|emple|transform|calcul|tom|recolect|colect|muestre|obtuv|"
    r"identific|cuantific|estim)\w*\b|(?:we|nosotros|nuestros?|nuestras?)\b)",
    re.IGNORECASE,
)
#
# Verbos en pasado con que un estudio narra lo que hizo u observó («se transformaron», «se realizó», «fueron»):
# una oración así, sin la que nombraba el estudio, se leería como parte del análisis propio.
#
PAST_NARRATION = re.compile(
    r"\b(?:[a-záéíúñ]+(?:aron|ieron)|[a-záéíúñ]+ó|fue|fueron|hubo|was|were|did|had)\b", re.IGNORECASE
)
ANAPHORIC_START = re.compile(
    r"^(?:esto|esta|este|ese|esa|eso|estos|estas|esos|esas|en\s+ese\s+caso|de\s+ser\s+as[ií]|this|these|that|those|"
    r"in\s+that\s+case|(?:los|las)\s+(?:m[aá]s|m[aá]xim|m[ií]nim|valores|concentraciones|niveles)|ambas?|dich[oa]s?|"
    r"the\s+(?:highest|lowest|values|concentrations|levels)|both|(?:en\s+)?otros?\s+(?:estudios?|trabajos?)|"
    r"another\s+study|por\s+su\s+parte|la\s+misma\s+autora|el\s+mismo\s+autor|"
    r"(?:el|la|los|las)\s+(?:primer[oa]s?|segund[oa]s?|tercer[oa]s?|[uú]ltim[oa]s?)(?=\s*[,;:]|"
    r"\s+(?:es|son|fue|fueron|era|eran|se|corresponde\w*|equivale\w*|representa\w*)\b)|"
    r"the\s+(?:former|latter)|the\s+(?:first|second|third|last)(?=\s*[,;:]|\s+(?:is|was|are|were)\b)|"
    r"(?:a|en|con|para|bajo|ante|desde|hasta|at|in|on|under|with|for|from)\s+"
    r"(?:este|esta|ese|esa|estos|estas|esos|esas|dich[oa]s?|tal(?:es)?|this|that|these|those|such)\b)\b",
    re.IGNORECASE,
)
GUIDELINE_CUES = re.compile(
    r"\b(?:gu[ií]as?|directrices?|l[ií]mites?|norma\w*|valor(?:es)?\s+de\s+referencia|recomend\w*|sugier\w*|suger\w*|"
    r"suggest\w*|m[aá]xim[oa]s?\s+permitid\w*|"
    r"niveles?\s+de\s+alerta|guideline\w*|limits?|standards?|criteri\w*|recommend\w*|alert\s+levels?|"
    r"permissible|threshold\w*|objective\w*)\b",
    re.IGNORECASE,
)
REFERENCE_LABELS = re.compile(
    r"\b(?:valor(?:es)?\s+(?:de\s+referencia|gu[ií]a)|nivel(?:es)?\s+(?:gu[ií]a|de\s+referencia)|umbral(?:es)?|"
    r"l[ií]mites?\s+(?:normativ\w*|permitid\w*|recomendad\w*|m[aá]xim\w*)|reference\s+values?|guideline\s+values?|"
    r"guidance\s+levels?|thresholds?)\b",
    re.IGNORECASE,
)
FREQUENCY_WORDS = re.compile(
    r"\b(?:habitual\w*|usual\w*|suelen?|sol[ií]an?|normalmente|generalmente|t[ií]pic\w*|frecuentemente|com[uú]nmente|"
    r"a\s+menudo|siempre|variabilidad\s+natural|usually|typically|typical|often|commonly|normally|generally|"
    r"frequently|always|natural\s+variability)\b",
    re.IGNORECASE,
)
INTERNAL_TERMS = re.compile(
    r"\b(?:ambig[üu]edad(?:es)?|ambiguit(?:y|ies)|extraction_warning|document_ids?|fact_ids?|rejected_draft)\b",
    re.IGNORECASE,
)
ORGANIZATION_NAMES = re.compile(
    r"\b(?:OMS|WHO|Organizaci[oó]n\s+Mundial\s+de\s+la\s+Salud|World\s+Health\s+Organi[sz]ation)\b"
)
ORGANIZATION_AGENT = re.compile(
    r"(?i:\b(?:by|por|seg[uú]n|according\s+to)\s+(?:(?:la|el|the)\s+)?)" + ORGANIZATION_NAMES.pattern
)
EVIDENCE_GAP = re.compile(
    r"\b(?:no\s+(?:se\s+)?(?:incluy\w*|figur\w*|aparec\w*|mencion\w*|especific\w*|contien\w*|indic\w*|aport\w*|"
    r"detall\w*|encontr\w*|describ\w*|informa\w*)|ning[uú]n\s+(?:fragmento|documento)|"
    r"(?:do|does)\s+not\s+(?:include|mention|specify|contain|state|report|provide)|not\s+(?:included|mentioned|specified|found))\b",
    re.IGNORECASE,
)
NEGATION = re.compile(r"\b(?:no|ni|sin|not|neither|without)\b", re.IGNORECASE)
STATED_PERCENTILE = re.compile(
    r"\b(?:\d{1,2}\s?(?:th|st|nd|rd|º|°|o)?\s?percentil\w*|percentil\w*\s?\d{1,2})\b",
    re.IGNORECASE,
)
LITERATURE_CLAUSE = re.compile(
    r"\b(?:bibliograf[ií]a|literatura|papers?|art[ií]culos\s+cient[ií]ficos|publicaciones|literature|"
    r"published\s+studies|scientific\s+(?:articles|papers))\b"
    r"(?:(?!\s(?:y|e|and)\s+(?:cu[aá]l|qu[eé]|c[oó]mo|cu[aá]nt|d[oó]nde|cu[aá]ndo|what|which|how|where|when)\b)"
    r"[^.;:?!¿,])*",
    re.IGNORECASE,
)
EVIDENCE_NOUNS = re.compile(
    r"\b(?:documentos?|fragmentos?|textos?|fuentes?|evidencia|bibliograf[ií]a|literatura|art[ií]culos?|"
    r"documents?|fragments?|texts?|sources?|evidence|literature|articles?)\b",
    re.IGNORECASE,
)
QUALIFIER_FAMILIES = {
    "atenuación": (
        re.compile(
            r"\b(?:podr[ií]an?|posibles?|posiblemente|(?:poco\s+|im)?probables?|probablemente|potencial(?:es|mente)?|quiz[aá]s?|"
            r"tal\s+vez|sugier\w*|suger\w*|parec\w*|se\s+asoci\w*|estar[ií]an?|ser[ií]an?|habr[ií]an?|"
            r"may|might|could|would|possibl[ey]|probabl[ey]|(?:un)?likely|potential(?:ly)?|suggest\w*|"
            r"appears?\s+to|seems?\s+to)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:podr[ií]an?|puede|pueden|pod[ií]a|posibles?|posiblemente|(?:poco\s+|im)?probables?|probablemente|potencial\w*|"
            r"quiz[aá]s?|tal\s+vez|sugier\w*|suger\w*|parec\w*|asoci\w*|relacion\w*|vincul\w*|suele|suelen|"
            r"estar[ií]an?|ser[ií]an?|habr[ií]an?|may|might|could|would|can|possibl[ey]|probabl[ey]|"
            r"(?:un)?likely|potential(?:ly)?|suggest\w*|appears?\s+to|seems?\s+to|associated|related)\b",
            re.IGNORECASE,
        ),
    ),
    "alcance del estudio": (
        re.compile(
            r"\b(?:en\s+(?:este|el\s+presente|nuestro)\s+(?:estudio|trabajo|an[aá]lisis|muestreo|relevamiento)|"
            r"en\s+esta\s+(?:investigaci[oó]n|tesis)|nuestros\s+(?:resultados|datos)|"
            r"in\s+(?:this|the\s+present|our)\s+(?:study|work|survey|research|analysis|investigation)|"
            r"our\s+(?:study|results|data|findings)|(?:during|throughout)\s+(?:this|the)\s+study|"
            r"the\s+(?:present|current)\s+study)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:estudios?|trabajos?|investigaci[oó]n(?:es)?|tesis|autor(?:es|as?)?|an[aá]lisis|monitoreos?|"
            r"muestreos?|campañas?|relevamientos?|seg[uú]n|de\s+acuerdo\s+con|et\s+al|reportaron|registraron|"
            r"observaron|encontraron|hallaron|midieron|detectaron|describieron|documentaron|informaron|"
            r"stud(?:y|ies)|work|research|authors?|survey|analysis|according\s+to|reported|found|observed|"
            r"measured|recorded)\b",
            re.IGNORECASE,
        ),
    ),
    "condición experimental": (
        re.compile(
            r"\b(?:laborator\w*|in\s+vitro|in\s+vivo|experimental\w*|bioensayos?|bioassays?|cultivos?|"
            r"cultures?|cultured|mesocosm\w*|microcosm\w*|ratones|ratas|mice|rats)\b",
            re.IGNORECASE,
        ),
    ),
    "cuantificador parcial": (
        re.compile(
            r"\b(?:some|certain|a\s+few|few|several|algun[oa]s|ciert[oa]s|poc[oa]s|vari[oa]s)\b",
            re.IGNORECASE,
        ),
    ),
    "ambiente marino": (
        re.compile(r"\b(?:marin[oa]s?|oce[aá]n\w*|agua\s+de\s+mar|marine|oceans?|oceanic|sea(?:water)?)\b", re.IGNORECASE),
    ),
    "ambiente costero": (
        re.compile(
            r"\b(?:coster[oa]s?|costas?|litoral(?:es)?|playas?|coastal|coasts?|beach(?:es)?|shores?|littoral)\b",
            re.IGNORECASE,
        ),
    ),
    "ambiente estuarino": (
        re.compile(r"\b(?:estuari[oa]s?|estuarin[oa]s?|estuar(?:y|ies|ine))\b", re.IGNORECASE),
    ),
    "agua dulce": (
        re.compile(
            r"\b(?:agua\s+dulce|aguas\s+dulces|dulceacu[ií]col\w*|freshwaters?|fresh\s+waters?)\b", re.IGNORECASE
        ),
    ),
}
SITUATED_FAMILIES = {"ambiente marino", "ambiente costero", "ambiente estuarino", "agua dulce"}
CAUSAL_LINKS = re.compile(
    r"\b(?:contribu(?:ye|yen|y[oó]|yeron|ir|ido|yendo)|favorec\w+|promuev\w+|promov\w+|propici(?:a|an|ó|aron|ar)|"
    r"desencaden\w+|conduc(?:e|en|ir|ido|ía|ían)\s+a|llev(?:a|an|ó|aron|ar)\s+a(?!\s+cabo)|provoc\w+|"
    r"caus(?:a|an|ó|aron|ado|ada|ados|adas|ar|ante|antes)|determin(?:a|an|ó|aron|ante|antes)|impuls\w+|"
    r"se\s+deb(?:e|en|i[oó]|ieron)\s+a|debid[oa]s?\s+a|a\s+causa\s+de|"
    r"origin(?:a|an|ó|aron|ado|ada|ados|adas|ar)|gener(?:a|an|ó|aron|ado|ada|ados|adas|ar)|"
    r"producid[oa]s?\s+por|atribu(?:ye|yen|ido|ida|idos|idas|ible|ibles|y[oó]|yeron)|"
    r"contribut(?:e|es|ed|ing)|favou?r(?:s|ed|ing)?|promot(?:e|es|ed|ing)|leads?\s+to|led\s+to|trigger\w*|"
    r"drives?|driven|induc(?:e|es|ed|ing)|caus(?:e|es|ed|ing)|due\s+to|because|"
    r"result(?:s|ed|ing)?\s+(?:in|from)|attribut(?:e|ed|able))\b",
    re.IGNORECASE,
)
RELATION_LANGUAGE = re.compile(
    rf"(?:{CAUSAL_LINKS.pattern})|\b(?:influ\w+|afect\w+|modul\w+|control\w+|regul\w+|depend\w+|"
    r"respond\w+|correlaci\w+|correlat\w+|asoci\w+|associat\w+|relacion\w+|relat(?:ed|ion\w*)|vincul\w+|"
    r"linked|conducive|propicias?|favorables?|favourable|consecuencia\w*|consequen\w+|"
    r"increas\w+|aument\w+|decreas\w+|disminu\w+)\b",
    re.IGNORECASE,
)
DETECTION_LIMIT = re.compile(
    r"\b(?:l[ií]mites?\s+de\s+(?:detecci[oó]n|cuantificaci[oó]n)|detection\s+limits?|"
    r"limits?\s+of\s+(?:detection|quantification)|quantification\s+limits?|LOD|LOQ)\b",
    re.IGNORECASE,
)
_EXPLANATION = (
    r"qu[eé]\s+(?:significa|implica|indica|representa|quiere\s+decir)|interpret\w*|explic\w*|por\s+qu[eé]|"
    r"a\s+qu[eé]\s+se\s+debe|what\s+does\s+(?:it|this|that)\s+mean|explain\w*|why"
)
EXPLANATION_CUES = re.compile(rf"\b(?:{_EXPLANATION})\b", re.IGNORECASE)
INTERPRETATION_CUES = re.compile(
    rf"\b(?:{_EXPLANATION}|es\s+(?:normal|alto|alta|bajo|baja|mucho|poco|preocupante|habitual|esperable|bueno|"
    r"malo)|compar\w*\s+con|en\s+comparaci[oó]n|contrast\w*|cotej\w*|respecto\s+(?:de|a)\s+(?:los|las|la|el)\s+"
    r"(?:valor|nivel|l[ií]mite|norma|gu[ií]a)\w*|is\s+(?:it|this|that)\s+(?:normal|high|low|a\s+lot)|"
    r"compare\w*\s+(?:with|to))\b",
    re.IGNORECASE,
)
PENDING_CONFIRMATION = re.compile(
    r"\b(?:confirm\w*|verific\w*|consultar|verify|verified)\b"
    r".{0,80}\b(?:unidad|fuente|dato|equivalencia|unit|source|original)\b",
    re.IGNORECASE,
)
CHEMICAL_NAME = re.compile(r"[-–]?(?:[A-Z][a-z]?[\d₀-₉]{0,3}[-–]?)+(?:[\d⁰¹²³⁴-⁹]?[+⁺−⁻])?")
ATTRIBUTED_SOURCE = re.compile(
    r"\b(?:[Ss]eg[uú]n|[Dd]e\s+acuerdo\s+(?:con|a)|[Cc]onforme\s+a|[Aa]ccording\s+to|[Aa]s\s+reported\s+by)\s+"
    r"(?:(?:la|el|lo|los|las|the)\s+)?"
    r"(?P<name>[A-ZÁÉÍÓÚÑ][\w-]*(?:\s+(?:(?:de|del|la|las|el|los|of|the|for)\s+)*[A-ZÁÉÍÓÚÑ][\w-]*)*)"
)
SOURCE_ALIASES = (("oms", "who", "world health", "organizacion mundial de la salud"),)
ATTRIBUTION_LEAD = re.compile(
    r"^(?:(?:seg[uú]n|de\s+acuerdo\s+(?:con|a)|conforme\s+a|according\s+to|as\s+reported\s+by)\b[^,]{0,160},\s*|"
    r"(?:(?:la|el|los|las|the)\s+)?[^,.;]{0,90}?\b(?:establece|establecen|indica|indican|señala|señalan|sostiene|"
    r"afirma|afirman|reporta|reportan|informa|informan|describe|describen|define|definen|plantea|plantean|"
    r"explica|explican|states?|indicates?|reports?|describes?|defines?|explains?)\s+(?:que|that)\s+)",
    re.IGNORECASE,
)
# Fechas escritas con números («01/09/2021», «2021-09-01»).
DATE_TOKEN = re.compile(r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{1,2}-\d{1,2})\b")
# Citas de autor y año («Sathicq et al. (2014)», «Pérez y colaboradores, 2010», «Gianuzzi (2011)», «(OMS, 2003)»):
# su año dice de qué obra es la afirmación, no es una cifra de lo que afirma.
CITATION_YEAR = re.compile(
    r"\b[A-ZÁÉÍÓÚÑ][\w'’-]+\s+(?:et\s+al\.?|y\s+col(?:aboradores|\.)|and\s+colleagues)(?:\s*,|\s+de|\s+in)?\s*\(?\s*"
    r"(?:19|20)\d{2}[a-z]?\)?"
    r"|\b[A-ZÁÉÍÓÚÑ][\w'’-]+\s*\(\s*(?:19|20)\d{2}[a-z]?\s*\)"
    r"|\(\s*[A-ZÁÉÍÓÚÑ][^()]{0,80}?,\s*(?:19|20)\d{2}[a-z]?\s*\)"
)
BOUND_WORDS = re.compile(
    r"\b(?:menor(?:es)?|inferior(?:es)?|mayor(?:es)?|superior(?:es)?|debajo|encima|menos\s+de|m[aá]s\s+de|hasta|"
    r"m[aá]xim[oa]s?|m[ií]nim[oa]s?|less|lower|below|under|greater|higher|above|over|more\s+than|up\s+to|"
    r"at\s+most|at\s+least)\b",
    re.IGNORECASE,
)
LAST_RESULT = re.compile(
    r"\b(?:ese|eso|esa|[uú]ltim[oa]|anterior|that|this|last|previous)\b", re.IGNORECASE
)
