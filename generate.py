"""
Synthetic crop disease Q&A generator.
Generates training data for a crop assistant TLM covering 10 African crops.
Pushes to HuggingFace every 1k examples. Fully resumable via state on HF.
"""

import os
import json
import time
import random
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from mistralai import Mistral
from huggingface_hub import HfApi

# ── Config ────────────────────────────────────────────────────────────────────
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
HF_TOKEN        = os.environ["HF_TOKEN"]
HF_REPO         = "rufatronics/crop-disease-tlm-data"
MODEL           = "mistral-small-2506"
BATCH_SIZE      = 10       # Q&A pairs per API call
PUSH_EVERY      = 1000     # push to HF every N examples
MAX_PER_RUN     = 50000    # cap per GitHub Actions run (~6h window)
TARGET_TOTAL    = 500000   # overall target across all runs
MAX_RETRIES     = 5
BASE_BACKOFF    = 2        # seconds, used as BASE_BACKOFF ** attempt

BUCKET_RATIOS = {
    "crop_knowledge": 0.85,
    "greetings":      0.10,
    "out_of_scope":   0.05,
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Knowledge base ────────────────────────────────────────────────────────────
CROP_KNOWLEDGE = {
    "cassava": {
        "disease": "cassava_mosaic",
        "facts": (
            "Cassava mosaic disease is caused by a virus spread by whiteflies. "
            "Infected leaves show yellow and green mosaic patterns, curling and distortion. "
            "Severely affected plants produce small, deformed tubers worth nothing at market. "
            "Use disease-free planting stems from healthy plants only — this is the number one rule. "
            "Remove and destroy infected plants immediately before the whitefly spreads it further. "
            "Control whiteflies with neem-based sprays or yellow sticky traps around the farm. "
            "Plant resistant varieties like TME 419, IITA TMS 30572 or any CMD-resistant variety. "
            "Do not replant stems from a farm that had mosaic — the virus travels with the stem."
        ),
        "practices": (
            "Plant cassava at the start of the rainy season for best establishment. "
            "Space plants 1 metre by 1 metre for good yield and ease of weeding. "
            "Weed regularly for the first 3 months — after that cassava shades out most weeds. "
            "Harvest at 9 to 18 months depending on variety and intended use. "
            "Mound the soil before planting to improve drainage and tuber size. "
            "Cassava can tolerate poor soils but responds well to potassium fertilizer."
        ),
    },
    "cocoa": {
        "disease": "cocoa_black_pod",
        "facts": (
            "Black pod disease is caused by Phytophthora palmivora and Phytophthora megakarya fungus. "
            "It appears as brown or black rotting patches on cocoa pods at any stage of development. "
            "Spreads extremely fast in wet, humid and shaded conditions — one infected pod can infect 10 others. "
            "Infected pods must be removed and buried deeply or burned — never leave them on the ground. "
            "Prune trees regularly to improve airflow and reduce the humidity inside the farm. "
            "Apply copper-based fungicide like Ridomil Gold or Nordox during the rainy season every 2 to 3 weeks. "
            "Can destroy up to one third of total harvest in a bad season if not managed. "
            "Harvest ripe pods every 2 weeks — overstay pods are the biggest source of infection."
        ),
        "practices": (
            "Prune cocoa trees after each harvest to allow light and air into the canopy. "
            "Mulch the base of trees with dead leaves to retain moisture and improve soil. "
            "Do not establish new farms too late in the rainy season. "
            "Remove diseased pods, mummified pods and deadwood in one go — this is called sanitation harvesting."
        ),
    },
    "cowpea": {
        "disease": "cowpea_aphid",
        "facts": (
            "Cowpea aphids are small shiny black or dark brown insects that cluster on young shoots, flowers and pods. "
            "They suck sap from the plant causing stunted growth, leaf curl, yellowing and wilting. "
            "Aphids also spread more than 30 plant viruses — so even a small infestation is a big threat. "
            "They produce sticky honeydew which encourages black sooty mould to grow on leaves blocking sunlight. "
            "Control with neem oil spray mixed with soap, garlic spray or pyrethrin-based insecticide. "
            "Remove and destroy heavily infested plant parts before aphids spread to the whole farm. "
            "Natural enemies like ladybird beetles, hoverflies and parasitic wasps feed on aphids — protect them. "
            "Chemical option: imidacloprid seed dressing or dimethoate foliar spray as last resort."
        ),
        "practices": (
            "Intercrop cowpea with maize at 4:1 ratio to reduce aphid pressure. "
            "Avoid planting downwind or next to already infested farms. "
            "Plant early to escape peak aphid populations which come in dry season. "
            "Cowpea fixes nitrogen in the soil — it improves fertility for the next crop planted on that land."
        ),
    },
    "maize": {
        "disease": "fall_armyworm",
        "facts": (
            "Fall armyworm is a caterpillar that hides deep inside the maize whorl and feeds mostly at night. "
            "Look for ragged irregular holes in leaves that look like windowpanes with a light through them. "
            "Wet sawdust-like frass inside the whorl is the clearest sign — that is the caterpillar droppings. "
            "One female moth lays up to 1000 eggs and the pest spreads across farms very fast in warm weather. "
            "Scout early morning or evening when caterpillars come to the surface — check 10-20 plants per field. "
            "Apply fine sand mixed with wood ash into the whorl — scratches and kills young caterpillars cheaply. "
            "Neem-based sprays work well on young caterpillars in the first 2 weeks of infestation. "
            "Chemical options: emamectin benzoate, spinetoram or chlorpyrifos — always follow label dosage. "
            "Intercropping maize with beans or soybean reduces fall armyworm attack significantly."
        ),
        "practices": (
            "Plant maize at the right time for your area to avoid peak moth migration season. "
            "Space at 75 cm between rows and 25 cm within rows, one seed per hole. "
            "Apply nitrogen fertilizer in two splits — half at planting, half at knee height. "
            "Harvest promptly when dry to avoid weevils and post-harvest losses in storage."
        ),
    },
    "groundnut": {
        "disease": "groundnut_rosette",
        "facts": (
            "Groundnut rosette is a deadly virus disease spread by the cowpea aphid Aphis craccivora. "
            "Infected plants are severely stunted — they stay small, bushy with yellowed or mottled bunched leaves. "
            "There is no cure once a plant is infected — it will never recover and must be removed immediately. "
            "The disease is most dangerous when it strikes early in the season — can cause total crop failure. "
            "Plant early at the very start of rains to escape peak aphid populations that come later. "
            "Use resistant varieties: SAMNUT 22, SAMNUT 23, RG 1 series or ask your extension officer for local ones. "
            "Control aphid vectors with insecticide seed treatment before planting. "
            "Weed the farm well — weeds host aphids which then move to groundnut plants."
        ),
        "practices": (
            "Plant groundnut in well-drained sandy loam soil — waterlogged soil causes pod rot. "
            "Space at 15 cm within rows and 30 cm between rows for good canopy closure against weeds. "
            "Groundnut fixes its own nitrogen — do not apply too much nitrogen or you get leaves not pods. "
            "Harvest before heavy rains arrive to avoid aflatoxin contamination in wet pods. "
            "Dry pods properly before storage — moisture above 9 percent causes aflatoxin which is very dangerous."
        ),
    },
    "mango": {
        "disease": "mango_anthracnose",
        "facts": (
            "Mango anthracnose is caused by the fungus Colletotrichum gloeosporioides. "
            "On leaves it causes dark brown irregular spots. On flowers it causes blossom blight killing flowers before fruit sets. "
            "On fruits it causes dark sunken lesions that expand rapidly after harvest as the fruit ripens. "
            "This is why mangoes that look fine on the tree can turn black within 2 days of harvest — the disease was already inside. "
            "Favoured by high humidity, frequent rain and temperatures between 24 and 32 degrees Celsius. "
            "Spray mancozeb or copper oxychloride at the flowering stage and repeat every 10 days during rains. "
            "Handle harvested fruits with extreme care — wounds and bruises let the fungus enter. "
            "Harvest fruits carefully, do not drop them, and store in a cool dry place immediately."
        ),
        "practices": (
            "Prune mango trees after every harvest to remove dead wood and improve air circulation. "
            "Remove fallen leaves and fruits from the ground regularly — they carry fungal spores. "
            "Avoid overhead irrigation which keeps foliage wet for too long. "
            "Harvest at correct maturity — a slight colour change at the tip is the sign for most varieties."
        ),
    },
    "plantain": {
        "disease": "plantain_bunchy_top",
        "facts": (
            "Banana bunchy top virus causes leaves to become short, stiff, narrow and crowded at the top of the plant. "
            "Infected leaves show dark green streaks and dots along the veins and leaf edges that turn yellow then brown. "
            "The plant stops growing properly and will never produce a bunch — it is a complete loss. "
            "Spread by banana aphids that carry the virus from infected to healthy plants. "
            "There is absolutely no cure for bunchy top — do not waste money on treatment. "
            "The moment you see symptoms: chop the infected plant to the ground and immediately inject the stem with kerosene or paraquat. "
            "This kills the corm underground so it cannot reshoot and keep hosting the virus. "
            "Never move suckers from a farm that had bunchy top — the virus travels inside the sucker with no visible signs. "
            "Plant only certified clean suckers from disease-free sources or tissue culture plantlets."
        ),
        "practices": (
            "Plant plantain suckers at the start of the rainy season for best establishment. "
            "Space at 3 metres by 3 metres to allow airflow between plants and ease harvesting. "
            "Desuckering is important — keep only 1 parent plant and 1 to 2 followers per mat at any time. "
            "Mulch heavily around plants to retain moisture and suppress weeds. "
            "Apply potassium-rich fertilizer — plantain is a heavy feeder, especially on potassium."
        ),
    },
    "rice": {
        "disease": "rice_blast",
        "facts": (
            "Rice blast is caused by the fungus Magnaporthe oryzae and is the most destructive rice disease in the world. "
            "On leaves it causes diamond-shaped or spindle-shaped lesions with a grey or white centre and brown or reddish edges. "
            "Neck blast attacks the base of the grain-bearing stalk causing it to fall over — this is the most damaging form. "
            "Disease thrives in cool nights, high humidity, heavy dew, and farms with too much nitrogen fertilizer. "
            "Never over-apply nitrogen fertilizer — excess nitrogen is like feeding the fungus directly. "
            "Spray tricyclazole, isoprothiolane or propiconazole fungicide at the first sign of leaf lesions. "
            "For neck blast prevention spray at boot stage before the panicle emerges — timing is everything. "
            "Plant blast-resistant varieties: FARO 52, FARO 57, NERICA series are good options in Nigeria and West Africa. "
            "Can reduce grain yield by 70 to 80 percent in severe epidemic years."
        ),
        "practices": (
            "Maintain proper water levels in paddy field — do not let the field dry out and flood alternately. "
            "Do not overcrowd seedlings during transplanting — proper spacing reduces disease pressure. "
            "Burn rice straw after harvest to destroy fungal spores that survive in crop residues. "
            "Use balanced NPK fertilizer, never more than the recommended nitrogen dose."
        ),
    },
    "tomato": {
        "disease": "tomato_blight",
        "facts": (
            "Tomato blight includes early blight caused by Alternaria solani and late blight caused by Phytophthora infestans. "
            "Early blight shows dark brown spots with concentric rings and yellow halo on older lower leaves — it starts from the bottom. "
            "Late blight causes irregular water-soaked greyish-green patches that turn brown and destroy entire plants within days in wet weather. "
            "Both diseases spread very fast in wet, cool and humid conditions — a week of rain can wipe out a whole farm. "
            "Remove and destroy infected leaves and plant parts at first sign — bag them and burn, do not compost. "
            "Spray mancozeb, chlorothalonil or copper-based fungicide every 7 days during rainy season as prevention. "
            "Avoid overhead watering — always water at the base of plants to keep foliage dry. "
            "Never plant tomato in the same plot two seasons in a row — rotate with maize or cowpea. "
            "Plant on raised beds or ridges to improve drainage and reduce soil splash onto lower leaves."
        ),
        "practices": (
            "Stake or cage tomato plants at 30 cm height to keep fruits and foliage off the ground. "
            "Space at 50 cm within rows and 75 cm between rows for good airflow. "
            "Apply balanced NPK at planting then top-dress with calcium nitrate at first flower. "
            "Harvest regularly every 2 to 3 days — leaving ripe fruits encourages pests and disease."
        ),
    },
}

CROPS = list(CROP_KNOWLEDGE.keys())

QUESTION_STYLES = [
    "worried_farmer",
    "direct_question",
    "symptom_description",
    "how_to_treat",
    "prevention",
    "identification",
    "urgency",
    "practice",
    "spread",
    "what_not_to_do",
    "comparison",           # is this worse than X
    "local_context",        # asking in African context, specific to their situation
    "timing",               # when to act, when to plant, when to harvest
    "cost_effective",       # cheap solutions, what a smallholder can actually afford
]

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are generating training data for an AI crop assistant called FarmBot that helps smallholder farmers in West and Central Africa.

LANGUAGE STYLE — CRITICAL:
- Use simple, practical African English. Farmers say "my maize is dying" not "the crop exhibits necrotic lesions"
- Warm and encouraging tone, like a knowledgeable neighbour not a scientist
- Short sentences. Direct advice. No jargon without a plain-English explanation right after
- Occasionally use natural African English phrases: "my friend", "first thing first", "the thing is", "no need to panic", "I tell you"
- Sometimes write questions in broken or pidgin English: "my tomato don spoil", "wetin dey cause this", "how I go treat am"
- Answers should feel spoken and practical, not like a textbook or report

ANSWER LENGTH RULES:
- Simple greetings or yes/no questions: 1-3 sentences
- Identification or what-is-this questions: 1 short paragraph
- How-to-treat or prevention questions: 1-2 paragraphs with clear steps
- Complex multi-part questions: up to 3 paragraphs, never more
- Never use bullet points or numbered lists in answers — write in natural flowing prose

OUTPUT FORMAT:
- Return ONLY a valid JSON array of objects, nothing else
- Each object has exactly two keys: "instruction" and "response"
- No markdown fences, no preamble, no explanation outside the JSON array
- instruction: the farmer's question (varied natural phrasing)
- response: FarmBot's answer following the length rules above

SCOPE:
- Only answer about these 10 crops: cassava, cocoa, cowpea, maize, groundnut, mango, plantain, rice, tomato
- For greetings bucket: respond warmly, briefly, and always mention what FarmBot can help with
- For out_of_scope bucket: politely decline and redirect to the 10 crops

ACCURACY:
- Base all facts strictly on the knowledge provided in each prompt
- Never invent chemical names, dosages, variety names or statistics not in the provided knowledge
- When mentioning chemicals always add: follow the label instructions carefully
- All advice must be applicable to smallholder African farmers with limited resources"""


def make_crop_prompt(crop: str, style: str, batch_size: int) -> str:
    kb = CROP_KNOWLEDGE[crop]
    return f"""Generate {batch_size} diverse Q&A pairs about {crop.upper()} farming for African smallholder farmers.

CROP: {crop.upper()}
DISEASE/PEST: {kb['disease'].replace('_', ' ')}

DISEASE/PEST FACTS TO USE:
{kb['facts']}

FARMING PRACTICES TO USE:
{kb['practices']}

QUESTION STYLE FOCUS: {style.replace('_', ' ')}

Generation rules:
- Vary phrasing heavily across all {batch_size} pairs — no two questions can sound alike
- Mix question styles even though {style} is the main focus
- Include at least 2 questions in pidgin or broken English style
- Some answers can reference local practices (burning infected material, using ash, neem)
- Answers must stay strictly within the facts above — do not add outside information
- Every answer should end with something actionable the farmer can do today or this week

Return a JSON array of exactly {batch_size} objects with "instruction" and "response" keys only."""


def make_greetings_prompt(batch_size: int) -> str:
    return f"""Generate {batch_size} diverse greeting and small-talk Q&A pairs for FarmBot.

FarmBot is a friendly AI crop assistant for African smallholder farmers covering: cassava, cocoa, cowpea, maize, groundnut, mango, plantain, rice, tomato.

Greeting types to cover across the {batch_size} pairs:
- Basic greetings: hello, hi, good morning, good afternoon, good evening
- How are you / how you dey
- Who are you / what are you
- What can you help me with / what do you know
- Thank you / you are helpful / God bless you
- I need help / I have a problem with my farm
- Pidgin versions: how body, how e dey, who you be

Rules:
- FarmBot responds warmly and briefly (2-3 sentences max)
- Always naturally mention 1-2 of the crops it can help with
- African English style: "Welcome my friend!", "I dey here to help you"
- Never give a long speech for a simple greeting
- Vary the responses significantly — no two answers should sound the same

Return a JSON array of exactly {batch_size} objects with "instruction" and "response" keys only."""


def make_out_of_scope_prompt(batch_size: int) -> str:
    topics = [
        "wheat", "yam", "potato", "soybean", "cotton", "sugarcane",
        "weather forecast", "fertilizer prices today", "bank loan",
        "human medicine", "animal disease", "goat sickness", "poultry disease",
        "politics", "football match", "how to cook jollof",
        "phone not charging", "how to make money online",
        "cassava price in the market", "exchange rate",
    ]
    selected = random.sample(topics, min(8, len(topics)))
    return f"""Generate {batch_size} Q&A pairs where farmers ask FarmBot things it CANNOT help with.

Topics to use (pick freely from this list): {', '.join(selected)}

Rules for FarmBot's responses:
- Politely explain it can only help with the 10 crops: cassava, cocoa, cowpea, maize, groundnut, mango, plantain, rice, tomato
- Warm, not cold or rude: "Ah my friend, that one is outside what I know..."
- Always end by offering to help with something it CAN do
- Keep decline responses SHORT — 2 sentences maximum
- Include some pidgin-style questions: "abeg how much be dollar today"
- Vary the phrasing of both questions and declines significantly

Return a JSON array of exactly {batch_size} objects with "instruction" and "response" keys only."""


# ── State management ──────────────────────────────────────────────────────────
def load_state(api: HfApi) -> dict:
    """Load generation state from HuggingFace or return fresh state."""
    try:
        path = api.hf_hub_download(
            repo_id=HF_REPO,
            filename="state.json",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        with open(path) as f:
            state = json.load(f)
        log.info(f"Resumed: {state['total_generated']:,} examples already generated")
        return state
    except Exception:
        log.info("No existing state, starting fresh")
        return {
            "total_generated": 0,
            "bucket_counts": {"crop_knowledge": 0, "greetings": 0, "out_of_scope": 0},
            "target": TARGET_TOTAL,
            "last_run": None,
            "runs_completed": 0,
        }


def save_state(api: HfApi, state: dict):
    """Push state.json to HuggingFace repo."""
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    tmp = "/tmp/state.json"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    try:
        api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo="state.json",
            repo_id=HF_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message=f"state update: {state['total_generated']:,} total examples",
        )
        log.info("State saved to HF")
    except Exception as e:
        log.warning(f"State save failed: {e}")


# ── API call with retry ───────────────────────────────────────────────────────
def call_mistral(client: Mistral, prompt: str, attempt: int = 0) -> list[dict] | None:
    """Call Mistral, parse JSON response, retry on any failure with backoff."""
    try:
        resp = client.chat.complete(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.92,
            max_tokens=4096,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if model adds them
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:])
            if raw.endswith("```"):
                raw = raw[:-3].strip()
        pairs = json.loads(raw)
        assert isinstance(pairs, list) and len(pairs) > 0
        for p in pairs:
            assert "instruction" in p and "response" in p
            assert isinstance(p["instruction"], str) and len(p["instruction"].strip()) > 5
            assert isinstance(p["response"], str) and len(p["response"].strip()) > 10
        return pairs

    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error attempt {attempt+1}: {e}")
    except AssertionError as e:
        log.warning(f"Validation error attempt {attempt+1}: {e}")
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "rate_limit" in err_str.lower() or "too many" in err_str.lower():
            wait = BASE_BACKOFF ** (attempt + 4)   # aggressive backoff for rate limits
            log.warning(f"Rate limit — waiting {wait}s before retry")
            time.sleep(wait)
        elif any(code in err_str[:4] for code in ["500", "502", "503", "504"]):
            wait = BASE_BACKOFF ** (attempt + 2)
            log.warning(f"Server error — waiting {wait}s: {e}")
            time.sleep(wait)
        else:
            log.error(f"API error attempt {attempt+1}: {e}")

    if attempt < MAX_RETRIES - 1:
        time.sleep(BASE_BACKOFF ** (attempt + 1))
        return call_mistral(client, prompt, attempt + 1)

    log.error("Max retries exceeded, skipping this batch")
    return None


# ── Quality filter ────────────────────────────────────────────────────────────
CROP_TERMS = set(CROPS + [
    "fall armyworm", "black pod", "mosaic", "blast", "blight",
    "anthracnose", "rosette", "aphid", "bunchy top", "armyworm",
    "whitefly", "phytophthora", "fungus", "disease", "pest",
    "harvest", "plant", "farm", "spray", "fertilizer", "yield",
])

def is_quality(item: dict, bucket: str) -> bool:
    q = item["instruction"].strip()
    a = item["response"].strip()
    # Basic length
    if len(q.split()) < 3:
        return False
    if len(a.split()) < 8:
        return False
    # Answer must not end with a question mark
    if a.rstrip().endswith("?"):
        return False
    # Answer must not just repeat the question
    if a.lower().strip()[:30] == q.lower().strip()[:30]:
        return False
    # Crop knowledge must reference at least one crop/disease term
    if bucket == "crop_knowledge":
        combined = (q + " " + a).lower()
        if not any(term in combined for term in CROP_TERMS):
            return False
    # Answer should not be a single word or empty
    if len(a.strip()) < 30:
        return False
    return True


# ── Push to HuggingFace as JSONL ─────────────────────────────────────────────
def push_batch_to_hf(api: HfApi, examples: list[dict], batch_index: int):
    """
    Append examples to HF repo as a numbered JSONL file.
    Each push is a separate file — avoids overwrite issues entirely.
    Files are named: data/batch_000001.jsonl, data/batch_000002.jsonl etc.
    HF datasets library will treat all JSONL files in data/ as one dataset.
    """
    filename = f"data/batch_{batch_index:06d}.jsonl"
    content = "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                     delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=filename,
        repo_id=HF_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message=f"add {len(examples)} examples → {filename}",
    )
    log.info(f"Pushed {len(examples)} examples → {filename}")
    Path(tmp_path).unlink(missing_ok=True)


# ── Bucket selector ───────────────────────────────────────────────────────────
def pick_bucket(state: dict) -> str:
    """Pick bucket with the biggest gap vs its target ratio."""
    total = max(state["total_generated"], 1)
    deficit = {}
    for bucket, ratio in BUCKET_RATIOS.items():
        target = ratio * TARGET_TOTAL
        current = state["bucket_counts"].get(bucket, 0)
        deficit[bucket] = target - current
    return max(deficit, key=deficit.get)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("Crop Disease TLM — Synthetic Data Generator")
    log.info("=" * 60)

    client = Mistral(api_key=MISTRAL_API_KEY)
    api    = HfApi()

    # Ensure repo exists
    try:
        api.create_repo(
            repo_id=HF_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            exist_ok=True,
        )
        log.info(f"HF repo ready → https://huggingface.co/datasets/{HF_REPO}")
    except Exception as e:
        log.error(f"Cannot access HF repo: {e}")
        raise

    state = load_state(api)

    if state["total_generated"] >= TARGET_TOTAL:
        log.info(f"Target of {TARGET_TOTAL:,} already reached!")
        return

    remaining = TARGET_TOTAL - state["total_generated"]
    log.info(f"Remaining: {remaining:,} examples to generate")

    generated_this_run = 0
    pending:     list[dict] = []
    batch_index: int = state.get("batch_index", 0)

    # Cycling iterators for variety
    crop_order  = CROPS * 100
    style_order = QUESTION_STYLES * 100
    random.shuffle(crop_order)
    random.shuffle(style_order)
    ci = si = 0

    while (
        generated_this_run < MAX_PER_RUN
        and state["total_generated"] < TARGET_TOTAL
    ):
        bucket = pick_bucket(state)

        if bucket == "crop_knowledge":
            crop   = crop_order[ci % len(crop_order)];  ci  += 1
            style  = style_order[si % len(style_order)]; si += 1
            prompt = make_crop_prompt(crop, style, BATCH_SIZE)
        elif bucket == "greetings":
            crop   = "general"
            prompt = make_greetings_prompt(BATCH_SIZE)
        else:
            crop   = "general"
            prompt = make_out_of_scope_prompt(BATCH_SIZE)

        pairs = call_mistral(client, prompt)
        if pairs is None:
            continue

        good = [p for p in pairs if is_quality(p, bucket)]
        if not good:
            log.warning("Batch failed quality filter entirely — skipping")
            continue

        for p in good:
            p["bucket"] = bucket
            p["crop"]   = crop

        pending.extend(good)
        state["bucket_counts"][bucket] = state["bucket_counts"].get(bucket, 0) + len(good)
        state["total_generated"] += len(good)
        generated_this_run       += len(good)

        log.info(
            f"run={generated_this_run:,}/{MAX_PER_RUN:,} | "
            f"total={state['total_generated']:,}/{TARGET_TOTAL:,} | "
            f"bucket={bucket} | good={len(good)}/{BATCH_SIZE}"
        )

        # Push every PUSH_EVERY examples
        if len(pending) >= PUSH_EVERY:
            to_push   = pending[:PUSH_EVERY]
            pending   = pending[PUSH_EVERY:]
            batch_index += 1
            try:
                push_batch_to_hf(api, to_push, batch_index)
                state["batch_index"] = batch_index
                save_state(api, state)
            except Exception as e:
                log.error(f"Push failed (batch {batch_index}): {e} — keeping in buffer")
                pending = to_push + pending   # put back, try again next cycle

        time.sleep(0.25)   # gentle rate limit cushion

    # Final push of anything remaining
    if pending:
        batch_index += 1
        log.info(f"Final push: {len(pending)} remaining examples → batch {batch_index:06d}")
        try:
            push_batch_to_hf(api, pending, batch_index)
            state["batch_index"] = batch_index
            save_state(api, state)
        except Exception as e:
            log.error(f"Final push failed: {e}")
            fallback = "/tmp/pending_fallback.jsonl"
            with open(fallback, "w", encoding="utf-8") as f:
                for item in pending:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            log.warning(f"Saved {len(pending)} examples to fallback: {fallback}")

    state["runs_completed"] = state.get("runs_completed", 0) + 1
    save_state(api, state)

    log.info("=" * 60)
    log.info(f"Run complete | this run: {generated_this_run:,} | total: {state['total_generated']:,}/{TARGET_TOTAL:,}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
