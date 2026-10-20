"""
Experiment configuration: system prompts and test items for the
Yoruba-system-prompt vs English-system-prompt comparison.
"""

# ---------------------------------------------------------------------------
# System prompt A: English instructions telling the model to respond in Yoruba
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_EN = """You are a helpful, knowledgeable assistant. The user will write to \
you in the Yoruba language. Always respond entirely in Yoruba. Do not switch to English \
or mix English words into your response unless a term has no natural Yoruba equivalent \
(e.g. a proper noun or brand name). Write fluent, natural, grammatically correct Yoruba \
that a native speaker would recognize as native-sounding, not a literal translation. Use \
Yoruba diacritics (underdots and tone marks) correctly where appropriate."""

# ---------------------------------------------------------------------------
# System prompt B: the same instructions, written natively in Yoruba
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_YO = """Ìwọ ni olùrànlọ́wọ́ tó ní ìmọ̀ jinlẹ̀ tí ó sì máa ń ràn àwọn èèyàn lọ́wọ́. \
Olùmúlò yóò kọ̀wé sí ọ ní èdè Yorùbá. Ìwọ gbọ́dọ̀ dá hun èsì rẹ ní èdè Yorùbá pátápátá. \
Má ṣe yí padà sí èdè Gẹ̀ẹ́sì tàbí kí o da ọ̀rọ̀ Gẹ̀ẹ́sì kún èsì rẹ, àyàfi bí ọ̀rọ̀ kan kò bá ní \
ìtúmọ̀ Yorùbá tó bá a mu (bí orúkọ ẹnìkan tàbí orúkọ ọjà kan). Kọ̀wé ní èdè Yorùbá tí ó tọ́, \
tí ó sì dán mọ́rán, gẹ́gẹ́ bí ẹni tí Yorùbá jẹ́ èdè abínibí rẹ̀ ṣe máa ń kọ̀wé, kì í ṣe bí \
ìtúmọ̀ tààràtà láti èdè mìíràn. Lo àwọn àmì àpèjúwe àti ohùn (bí ẹ, ọ, ṣ, àti àmì ohùn) \
ní ibi tí ó yẹ."""

# ---------------------------------------------------------------------------
# Test items: diverse Yoruba-language tasks, same input used for both arms.
# `category` groups items by task type for later breakdown.
# ---------------------------------------------------------------------------
# `reference` (optional): ground-truth material used by the LLM-judge to
# score faithfulness/groundedness (see llm_judge.py). For factual_qa and
# reasoning items it's the correct answer; for summarization it's the
# source text the summary must stay faithful to. Items with no `reference`
# are judged only for internal consistency / not fabricating claims beyond
# what the prompt supports (faithfulness still applies; groundedness in the
# strict "grounded in a source doc" sense mostly doesn't).
TEST_ITEMS = [
    {
        "id": "qa_factual_1",
        "category": "factual_qa",
        "input": "Kí ni olú-ìlú Nàìjíríà, kí sì ni ìdí tí wọ́n fi yí i padà láti Èkó sí Abuja?",
        "reference": (
            "The capital of Nigeria is Abuja. The capital was moved from Lagos to Abuja "
            "in 1991. Reasons commonly cited: Abuja is centrally located within Nigeria "
            "(more neutral/accessible to all ethnic and regional groups than Lagos, which "
            "is on the coast and associated with the Yoruba southwest), Lagos was severely "
            "overcrowded, and Abuja was a planned city built specifically to serve as the "
            "federal capital."
        ),
    },
    {
        "id": "advice_1",
        "category": "advice",
        "input": "Mo fẹ́ bẹ̀rẹ̀ iṣẹ́ àṣàrò-ọwọ́ kékeré ní ìlú mi. Kí ni àwọn ìmọ̀ràn mẹ́ta tí o lè fún mi tí yóò ràn mí lọ́wọ́ láti ṣàṣeyọrí?",
    },
    {
        "id": "explain_1",
        "category": "explanation",
        "input": "Ṣàlàyé bí òjò ṣe ń rọ̀ fún ọmọdé kan tí kò tí ì mọ̀ nípa ìmọ̀ sáyẹ́nsì rárá.",
        "reference": (
            "Rain forms when the sun heats water in oceans, rivers, and lakes, causing it "
            "to evaporate into water vapor that rises into the air. As it rises and cools, "
            "the vapor condenses around tiny particles into water droplets, forming clouds. "
            "When enough droplets combine and become heavy enough, gravity pulls them down "
            "as rain."
        ),
    },
    {
        "id": "story_1",
        "category": "creative",
        "input": "Kọ ìtàn kékeré kan (nǹkan bí ẹ̀ẹ́dẹ́gbẹ̀ta ọ̀rọ̀) nípa ọkùnrin kan tí ó ṣàwárí ohun àràmàǹdà nínú igbó.",
    },
    {
        "id": "opinion_1",
        "category": "reasoning",
        "input": "Kí ni èrò rẹ nípa lílo fóònù alágbèéká láàrin àwọn ọmọ ilé-ìwé? Ṣé ó dára tàbí ó burú? Ṣàlàyé ìdí.",
    },
    {
        "id": "instruction_1",
        "category": "task_instruction",
        "input": "Kọ àkọsílẹ̀ ìmúrasílẹ̀ oúnjẹ jollof rice ní ọ̀nà tí ó rọrùn láti tẹ̀lé, ní ìṣísẹ̀-n-tẹ̀lé.",
    },
    {
        "id": "summarize_1",
        "category": "summarization",
        "input": "Ṣe àkótán ọ̀rọ̀ yìí ní gbolohun mẹ́ta pàápàá: 'Ẹ̀kọ́ jẹ́ ọ̀nà kan pàtàkì láti mú ìdàgbàsókè bá orílẹ̀-èdè èyíkéyìí. Nígbà tí àwọn ọmọdé bá kàwé dáradára, wọ́n máa ń ní àǹfààní láti rí iṣẹ́ tí ó dára, kí wọ́n sì lè kópa nínú ìdàgbàsókè agbègbè wọn. Ṣùgbọ́n ní ọ̀pọ̀ ìgbà, àìní owó àti amáyédẹrùn tí ó dára ń dí àwọn ọmọdé lọ́wọ́ láti kàwé dáradára.'",
        "reference": (
            "Source text (Yoruba, to be summarized faithfully — the summary must not "
            "introduce claims absent from it): 'Education is a key way to bring "
            "development to any country. When children are well educated, they get "
            "better access to good jobs and can participate in developing their "
            "communities. But often, lack of money and good facilities prevents "
            "children from being well educated.'"
        ),
    },
    {
        "id": "conversation_1",
        "category": "casual_chat",
        "input": "Bàwo ni o ṣe rí lónìí? Sọ báwo ni ọjọ́ rẹ ti rí fún mi, kí o sì bí mi lẹ́ẹ̀ẹ́kan náà nípa ọjọ́ tèmi.",
    },
    {
        "id": "reasoning_1",
        "category": "reasoning",
        "input": "Bí ọkọ̀ ojú irin kan bá ń rìn ní kìlómítà 60 ní wákàtí kan, yóò ti rìn kìlómítà mélòó ní wákàtí márùn-ún àbọ̀? Ṣàlàyé bí o ṣe rí ìdáhùn náà.",
        "reference": "Correct answer: 60 km/h x 5.5 hours = 330 kilometers.",
    },
    {
        "id": "task_translation_1",
        "category": "task_instruction",
        "input": "Fún mi ní àwọn ọ̀nà márùn-ún tí mo lè gbà tọ́jú ìlera mi dáradára ní ilé, láìsí lílọ sí ilé-ìwòsàn.",
    },
]
