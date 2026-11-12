
"""Léxico de entidades para o GraphRAG Med.

Responsabilidades:
- Reconhecimento de entidades por dicionário.
- Resolução de sinônimos.
- Normalização de acentos sem perder os offsets do texto original.
- Reconhecimento de medicamentos, condições, classes e efeitos adversos.
- Inclusão de medicamentos e classes pelos metadados do ChatBulário.

Os identificadores são locais ao projeto. Para uso clínico em produção,
recomenda-se alinhamento futuro com terminologias como RxNorm e MeSH.
"""

import re
import unicodedata
from dataclasses import dataclass


# ============================================================
# ENTIDADES
# ============================================================

# Formato:
# (id canônico, tipo, nome, sinônimos)

ENTITIES = [
    # --------------------------------------------------------
    # MEDICAMENTOS
    # --------------------------------------------------------

    ("DRUG:warfarin", "Drug", "Warfarin", ["Coumadin"]),
    ("DRUG:aspirin", "Drug", "Aspirin", ["acetylsalicylic acid"]),
    ("DRUG:amiodarone", "Drug", "Amiodarone", []),
    ("DRUG:lisinopril", "Drug", "Lisinopril", []),
    ("DRUG:spironolactone", "Drug", "Spironolactone", []),
    ("DRUG:ibuprofen", "Drug", "Ibuprofen", ["Advil", "Motrin"]),
    ("DRUG:propranolol", "Drug", "Propranolol", []),
    ("DRUG:verapamil", "Drug", "Verapamil", []),
    ("DRUG:metformin", "Drug", "Metformin", []),
    ("DRUG:simvastatin", "Drug", "Simvastatin", []),

    # --------------------------------------------------------
    # CONDIÇÕES
    # --------------------------------------------------------

    (
        "COND:hypertension",
        "Condition",
        "Hypertension",
        ["high blood pressure", "hipertensão", "pressão alta"],
    ),
    (
        "COND:heart_failure",
        "Condition",
        "Heart failure",
        ["insuficiência cardíaca"],
    ),
    (
        "COND:atrial_fibrillation",
        "Condition",
        "Atrial fibrillation",
        ["fibrilação atrial"],
    ),
    (
        "COND:dvt",
        "Condition",
        "Deep vein thrombosis",
        ["DVT", "trombose venosa profunda"],
    ),
    (
        "COND:angina",
        "Condition",
        "Angina",
        [],
    ),
    (
        "COND:asthma",
        "Condition",
        "Asthma",
        ["asma"],
    ),
    (
        "COND:t2d",
        "Condition",
        "Type 2 diabetes",
        [
            "T2DM",
            "diabetes tipo 2",
            "diabetes mellitus tipo 2",
        ],
    ),
    (
        "COND:renal_failure",
        "Condition",
        "Renal failure",
        ["kidney failure", "insuficiência renal"],
    ),
    (
        "COND:hypercholesterolemia",
        "Condition",
        "Hypercholesterolemia",
        ["hipercolesterolemia"],
    ),
    (
        "COND:pain",
        "Condition",
        "Pain",
        ["dor"],
    ),
    (
        "COND:mi",
        "Condition",
        "Myocardial infarction",
        [
            "heart attack",
            "infarto",
            "infarto do miocárdio",
        ],
    ),
    (
        "COND:angioedema",
        "Condition",
        "Angioedema",
        [],
    ),
    (
        "COND:peptic_ulcer",
        "Condition",
        "Peptic ulcer",
        ["úlcera péptica"],
    ),

    # --------------------------------------------------------
    # EFEITOS ADVERSOS EXISTENTES
    # --------------------------------------------------------

    (
        "AE:hemorrhage",
        "AdverseEffect",
        "Hemorrhage",
        [
            "bleeding",
            "haemorrhage",
            "hemorragia",
            "sangramento",
        ],
    ),
    (
        "AE:gi_bleeding",
        "AdverseEffect",
        "Gastrointestinal bleeding",
        [
            "GI bleeding",
            "sangramento gastrointestinal",
        ],
    ),
    (
        "AE:hyperkalemia",
        "AdverseEffect",
        "Hyperkalemia",
        [
            "hipercalemia",
            "hiperpotassemia",
        ],
    ),
    (
        "AE:cough",
        "AdverseEffect",
        "Cough",
        ["tosse"],
    ),
    (
        "AE:bradycardia",
        "AdverseEffect",
        "Bradycardia",
        ["bradicardia"],
    ),
    (
        "AE:lactic_acidosis",
        "AdverseEffect",
        "Lactic acidosis",
        ["acidose láctica"],
    ),
    (
        "AE:myopathy",
        "AdverseEffect",
        "Myopathy",
        ["miopatia"],
    ),
    (
        "AE:rhabdomyolysis",
        "AdverseEffect",
        "Rhabdomyolysis",
        ["rabdomiólise"],
    ),

    # --------------------------------------------------------
    # EFEITOS ADVERSOS GASTROINTESTINAIS
    # --------------------------------------------------------

    (
        "AE:nausea",
        "AdverseEffect",
        "Nausea",
        [
            "náusea",
            "náuseas",
            "enjoo",
            "enjoos",
        ],
    ),
    (
        "AE:vomiting",
        "AdverseEffect",
        "Vomiting",
        [
            "vômito",
            "vômitos",
            "vomito",
            "vomitos",
            "emese",
        ],
    ),
    (
        "AE:constipation",
        "AdverseEffect",
        "Constipation",
        [
            "constipação",
            "prisão de ventre",
        ],
    ),
    (
        "AE:diarrhea",
        "AdverseEffect",
        "Diarrhea",
        [
            "diarreia",
            "diarréia",
        ],
    ),
    (
        "AE:abdominal_pain",
        "AdverseEffect",
        "Abdominal pain",
        [
            "dor abdominal",
            "dores abdominais",
            "abdominal discomfort",
            "desconforto abdominal",
        ],
    ),
    (
        "AE:dry_mouth",
        "AdverseEffect",
        "Dry mouth",
        [
            "boca seca",
            "secura da boca",
            "xerostomia",
        ],
    ),

    # --------------------------------------------------------
    # EFEITOS ADVERSOS NEUROLÓGICOS
    # --------------------------------------------------------

    (
        "AE:dizziness",
        "AdverseEffect",
        "Dizziness",
        [
            "tontura",
            "tonturas",
            "vertigem",
            "vertigo",
        ],
    ),
    (
        "AE:somnolence",
        "AdverseEffect",
        "Somnolence",
        [
            "sonolência",
            "sonolencia",
            "sedação",
            "sedacao",
            "drowsiness",
        ],
    ),
    (
        "AE:headache",
        "AdverseEffect",
        "Headache",
        [
            "dor de cabeça",
            "cefaleia",
            "cefaléia",
            "head pain",
        ],
    ),
    (
        "AE:confusion",
        "AdverseEffect",
        "Confusion",
        [
            "confusão",
            "confusao",
            "estado confusional",
        ],
    ),
    (
        "AE:seizure",
        "AdverseEffect",
        "Seizure",
        [
            "convulsão",
            "convulsões",
            "crise convulsiva",
            "crises convulsivas",
        ],
    ),
    (
        "AE:insomnia",
        "AdverseEffect",
        "Insomnia",
        [
            "insônia",
            "insonia",
            "dificuldade para dormir",
        ],
    ),

    # --------------------------------------------------------
    # EFEITOS ADVERSOS RESPIRATÓRIOS E CARDIOVASCULARES
    # --------------------------------------------------------

    (
        "AE:respiratory_depression",
        "AdverseEffect",
        "Respiratory depression",
        [
            "depressão respiratória",
            "depressao respiratoria",
            "respiração lenta",
            "respiracao lenta",
        ],
    ),
    (
        "AE:hypotension",
        "AdverseEffect",
        "Hypotension",
        [
            "hipotensão",
            "hipotensao",
            "pressão baixa",
            "pressao baixa",
        ],
    ),
    (
        "AE:palpitations",
        "AdverseEffect",
        "Palpitations",
        [
            "palpitação",
            "palpitações",
            "palpitacao",
            "palpitacoes",
        ],
    ),

    # --------------------------------------------------------
    # OUTROS EFEITOS ADVERSOS
    # --------------------------------------------------------

    (
        "AE:fatigue",
        "AdverseEffect",
        "Fatigue",
        [
            "fadiga",
            "cansaço",
            "cansaco",
            "astenia",
        ],
    ),
    (
        "AE:pruritus",
        "AdverseEffect",
        "Pruritus",
        [
            "prurido",
            "coceira",
            "itching",
        ],
    ),
    (
        "AE:rash",
        "AdverseEffect",
        "Rash",
        [
            "erupção cutânea",
            "erupcao cutanea",
            "exantema",
            "rash cutâneo",
            "rash cutaneo",
        ],
    ),
    (
        "AE:sweating",
        "AdverseEffect",
        "Sweating",
        [
            "sudorese",
            "transpiração",
            "transpiracao",
        ],
    ),
    (
        "AE:serotonin_syndrome",
        "AdverseEffect",
        "Serotonin syndrome",
        [
            "síndrome serotoninérgica",
            "sindrome serotoninergica",
        ],
    ),

    # --------------------------------------------------------
    # CLASSES TERAPÊUTICAS
    # --------------------------------------------------------

    (
        "CLASS:anticoagulant",
        "DrugClass",
        "Anticoagulant",
        ["anticoagulante"],
    ),
    (
        "CLASS:ace_inhibitor",
        "DrugClass",
        "ACE inhibitor",
        ["inibidor da ECA"],
    ),
    (
        "CLASS:beta_blocker",
        "DrugClass",
        "Beta blocker",
        ["betabloqueador", "beta-bloqueador"],
    ),
    (
        "CLASS:biguanide",
        "DrugClass",
        "Biguanide",
        ["biguanida"],
    ),
    (
        "CLASS:statin",
        "DrugClass",
        "Statin",
        ["estatina"],
    ),
    (
        "CLASS:nsaid",
        "DrugClass",
        "NSAID",
        [
            "AINE",
            "anti-inflamatório não esteroidal",
            "anti-inflamatorio nao esteroidal",
        ],
    ),
    (
        "CLASS:antiarrhythmic",
        "DrugClass",
        "Antiarrhythmic",
        ["antiarrítmico", "antiarritmico"],
    ),
    (
        "CLASS:aldosterone_antagonist",
        "DrugClass",
        "Aldosterone antagonist",
        ["antagonista da aldosterona"],
    ),
    (
        "CLASS:ccb",
        "DrugClass",
        "Calcium channel blocker",
        ["bloqueador dos canais de cálcio"],
    ),
]


# ============================================================
# MENÇÃO ENCONTRADA NO TEXTO
# ============================================================

@dataclass(frozen=True)
class Mention:
    start: int
    end: int
    id: str
    type: str
    text: str


# ============================================================
# LÉXICO
# ============================================================

class Lexicon:
    """Reconhecimento de entidades por vocabulário e sinônimos."""

    def __init__(self, entities=ENTITIES):
        self.by_id = {}
        self.surface = {}
        self.rx = None

        for eid, etype, name, synonyms in entities:
            self.add_entity(
                {
                    "id": eid,
                    "type": etype,
                    "name": name,
                    "synonyms": synonyms,
                },
                compile_now=False,
            )

        self._compile()

    # --------------------------------------------------------
    # CONSTRUÇÃO PARA O CHATBULÁRIO
    # --------------------------------------------------------

    @classmethod
    def for_chatbulario(cls):
        """Cria o léxico base de condições e efeitos adversos.

        Os medicamentos e classes do dataset são adicionados
        posteriormente por extend_from_docs().
        """

        base = [
            entity
            for entity in ENTITIES
            if entity[1] in {
                "Condition",
                "AdverseEffect",
            }
        ]

        return cls(entities=base)

    # --------------------------------------------------------
    # NORMALIZAÇÃO
    # --------------------------------------------------------

    @staticmethod
    def _normalize(value):
        """Normaliza texto para comparação sem diferenciar acentos."""

        value = unicodedata.normalize(
            "NFKD",
            str(value or ""),
        )

        value = "".join(
            char
            for char in value
            if not unicodedata.combining(char)
        )

        return value.casefold()

    @staticmethod
    def _normalize_with_map(text):
        """Normaliza o texto preservando o mapeamento dos offsets.

        O resultado contém:
        - Texto normalizado.
        - Mapa entre cada caractere normalizado e sua posição
          correspondente no texto original.

        Isso permite reconhecer, por exemplo, 'náusea' e 'nausea'
        sem devolver offsets incorretos ao pipeline.
        """

        original = str(text or "")

        normalized_chars = []
        position_map = []

        for original_index, char in enumerate(original):

            normalized_char = unicodedata.normalize(
                "NFKD",
                char,
            )

            normalized_char = "".join(
                item
                for item in normalized_char
                if not unicodedata.combining(item)
            ).casefold()

            for item in normalized_char:
                normalized_chars.append(item)
                position_map.append(original_index)

        return (
            "".join(normalized_chars),
            position_map,
        )

    # --------------------------------------------------------
    # IDENTIFICADORES
    # --------------------------------------------------------

    @staticmethod
    def _slug(value):
        """Gera identificador estável compatível com o pipeline."""

        normalized = unicodedata.normalize(
            "NFKD",
            str(value or ""),
        )

        normalized = normalized.encode(
            "ascii",
            "ignore",
        ).decode("ascii")

        return (
            re.sub(
                r"[^a-z0-9]+",
                "_",
                normalized.lower(),
            ).strip("_")
            or "unknown"
        )

    # --------------------------------------------------------
    # CADASTRO DE ENTIDADES
    # --------------------------------------------------------

    def add_entity(self, entity, compile_now=True):
        """Adiciona ou atualiza uma entidade e seus sinônimos."""

        eid = str(entity["id"])
        etype = str(entity["type"])

        name = str(
            entity.get("name") or eid
        ).strip()

        synonyms = [
            str(item).strip()
            for item in entity.get("synonyms", [])
            if str(item).strip()
        ]

        if etype not in {
            "Drug",
            "Condition",
            "AdverseEffect",
            "DrugClass",
            "Procedure",
            "Anatomy",
        }:
            raise ValueError(
                "Tipo de entidade inválido: {}".format(etype)
            )

        if eid in self.by_id:

            old = self.by_id[eid]

            if old["type"] != etype:
                raise ValueError(
                    "O identificador {} já existe com tipo {}.".format(
                        eid,
                        old["type"],
                    )
                )

            synonyms = list(
                dict.fromkeys(
                    old["synonyms"] + synonyms
                )
            )

            if len(name) > len(old["name"]):
                old["name"] = name

            old["synonyms"] = synonyms

        else:

            self.by_id[eid] = {
                "id": eid,
                "type": etype,
                "name": name,
                "synonyms": synonyms,
            }

        forms = [
            self.by_id[eid]["name"],
            *synonyms,
        ]

        for form in forms:

            normalized = self._normalize(form)

            if len(normalized) < 3:
                continue

            # Mantém a primeira entidade registrada em caso de
            # colisão de sinônimos. Isso evita que uma entidade
            # posterior substitua silenciosamente outra.
            self.surface.setdefault(
                normalized,
                eid,
            )

        if compile_now:
            self._compile()

    # --------------------------------------------------------
    # EXTENSÃO COM METADADOS DO CHATBULÁRIO
    # --------------------------------------------------------

    def extend_from_docs(self, docs):
        """Adiciona medicamentos e classes dos metadados."""

        for doc in docs or []:

            meta = doc.get("metadata") or {}

            active = str(
                meta.get("principio_ativo")
                or meta.get("principio_ativo_csv")
                or ""
            ).strip()

            product = str(
                meta.get("nome_produto")
                or meta.get("nome_comercial")
                or ""
            ).strip()

            canonical = active or product

            if canonical:

                eid = "DRUG:" + self._slug(canonical)

                aliases = [
                    product,
                    str(
                        meta.get("nome_comercial") or ""
                    ).strip(),
                ]

                # Remove sais e termos de formulação comuns
                # para reconhecer a substância pelo nome curto.
                salt_terms = {
                    "cloridrato",
                    "maleato",
                    "succinato",
                    "fumarato",
                    "bromidrato",
                    "sodica",
                    "sódica",
                    "potassica",
                    "potássica",
                    "calcica",
                    "cálcica",
                    "magnesica",
                    "magnésica",
                    "de",
                    "do",
                    "da",
                }

                core = " ".join(
                    word
                    for word in canonical.split()
                    if self._normalize(word)
                    not in {
                        self._normalize(term)
                        for term in salt_terms
                    }
                )

                if (
                    core
                    and self._normalize(core)
                    != self._normalize(canonical)
                ):
                    aliases.append(core)

                if core and len(core.split()) > 1:
                    aliases.append(
                        core.split()[-1]
                    )

                clean_aliases = list(
                    dict.fromkeys(
                        alias
                        for alias in aliases
                        if alias
                        and self._normalize(alias)
                        != self._normalize(canonical)
                    )
                )

                self.add_entity(
                    {
                        "id": eid,
                        "type": "Drug",
                        "name": canonical,
                        "synonyms": clean_aliases,
                    },
                    compile_now=False,
                )

            therapeutic_class = str(
                meta.get("classe_terapeutica") or ""
            ).strip()

            if canonical and therapeutic_class:

                cid = (
                    "CLASS:"
                    + self._slug(therapeutic_class)
                )

                self.add_entity(
                    {
                        "id": cid,
                        "type": "DrugClass",
                        "name": therapeutic_class,
                        "synonyms": [],
                    },
                    compile_now=False,
                )

        self._compile()

    # --------------------------------------------------------
    # COMPILAÇÃO DO RECONHECEDOR
    # --------------------------------------------------------

    def _compile(self):
        """Compila expressões regulares com formas normalizadas."""

        forms = sorted(
            self.surface.keys(),
            key=len,
            reverse=True,
        )

        if not forms:
            self.rx = re.compile(r"(?!)")
            return

        alternatives = "|".join(
            re.escape(form)
            for form in forms
        )

        self.rx = re.compile(
            r"(?<!\w)(?:"
            + alternatives
            + r")(?!\w)",
            re.UNICODE,
        )

    # --------------------------------------------------------
    # RECONHECIMENTO DE MENÇÕES
    # --------------------------------------------------------

    def find(self, text):
        """Localiza entidades no texto original.

        A comparação é feita sem diferenciar acentos,
        mas os offsets e o texto retornado correspondem
        exatamente ao texto original.
        """

        original = str(text or "")

        if not original or self.rx is None:
            return []

        normalized, position_map = (
            self._normalize_with_map(original)
        )

        if not normalized or not position_map:
            return []

        mentions = []

        for match in self.rx.finditer(normalized):

            normalized_form = match.group()
            eid = self.surface.get(normalized_form)

            if eid is None:
                continue

            normalized_start = match.start()
            normalized_end = match.end()

            if normalized_start >= len(position_map):
                continue

            if normalized_end <= 0:
                continue

            original_start = position_map[
                normalized_start
            ]

            original_end = (
                position_map[normalized_end - 1] + 1
            )

            entity = self.by_id[eid]

            mentions.append(
                Mention(
                    start=original_start,
                    end=original_end,
                    id=eid,
                    type=entity["type"],
                    text=original[original_start:original_end],
                )
            )

        return mentions

    # --------------------------------------------------------
    # EXPORTAÇÃO DAS ENTIDADES
    # --------------------------------------------------------

    def entities(self):
        """Retorna todas as entidades cadastradas."""

        return list(
            self.by_id.values()
        )
