import os
import time
import re
import unicodedata
import json
import hashlib
from typing import List, Tuple, Optional

import redis  # ✅ Redis

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

from langchain.prompts import PromptTemplate


# -----------------------------
# CONFIG
# -----------------------------
PDF_PATH = "data/doc.pdf"
INDEX_DIR = "storage/faiss_doc_index"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"
TEMPERATURE = 0

TOP_K = 4

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

NO_ANSWER_TEXT = "Bu bilgi dokümanda yok."

BM25_WEIGHT = 0.6
FAISS_WEIGHT = 0.4

MIN_DOCS = 1
MAX_REWRITE_WORDS = 16

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

ANSWER_TTL_SEC = int(os.getenv("ANSWER_TTL_SEC", "600"))  # 10 dk

MEM_TTL_SEC = int(os.getenv("MEM_TTL_SEC", "86400"))      # 1 gün
MEM_MAX_TURNS = int(os.getenv("MEM_MAX_TURNS", "40"))
MEM_TAIL_FOR_CACHE = int(os.getenv("MEM_TAIL_FOR_CACHE", "6"))
MEM_TURNS_FOR_PROMPT = int(os.getenv("MEM_TURNS_FOR_PROMPT", "12"))


# -----------------------------
# PROMPTS
# -----------------------------
ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question", "no_answer_text", "chat_history"],
    template=(
        "Sen bir doküman asistanısın.\n"
        "SADECE aşağıdaki CONTEXT içindeki bilgilere dayanarak cevap ver.\n"
        "Eğer cevap CONTEXT içinde yoksa, tam olarak şunu yaz: {no_answer_text}\n"
        "Ek kaynak/link/tarih yazma. Kısa ve net cevap ver.\n\n"
        "CHAT HISTORY (kısa):\n{chat_history}\n\n"
        "CONTEXT:\n{context}\n\n"
        "SORU:\n{question}\n\n"
        "CEVAP:"
    ),
)

REWRITE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "Görev: Kullanıcı sorusunu doküman araması için kısa bir arama sorgusuna çevir.\n"
        "Kurallar:\n"
        "- Türkçe yaz.\n"
        f"- En fazla {MAX_REWRITE_WORDS} kelime.\n"
        "- Sadece anahtar kelimeler ve gerekiyorsa sayılar.\n"
        "- Noktalama/emoji/alıntı işareti kullanma.\n"
        "- Tek satır.\n\n"
        "Soru: {question}\n"
        "Arama sorgusu:"
    ),
)

FALLBACK_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "Görev: Kullanıcının sorusu için, dokümanda aramak üzere 3 alternatif kısa arama sorgusu öner.\n"
        "Kurallar:\n"
        "- Türkçe yaz.\n"
        "- Her satırda 1 sorgu.\n"
        "- Her sorgu 4-10 kelime.\n"
        "- Noktalama kullanma.\n\n"
        "Soru: {question}\n"
        "Alternatif sorgular:"
    ),
)


# -----------------------------
# REDIS CLIENT
# -----------------------------
def get_redis() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


r = get_redis()


# -----------------------------
# CHAT MEMORY
# -----------------------------
def chat_key(session_id: str) -> str:
    return f"chat:{session_id}"


def add_message(session_id: str, role: str, content: str) -> None:
    item = {"role": role, "content": content, "ts": int(time.time())}
    key = chat_key(session_id)
    r.rpush(key, json.dumps(item, ensure_ascii=False))
    r.expire(key, MEM_TTL_SEC)
    r.ltrim(key, -MEM_MAX_TURNS, -1)


def get_history(session_id: str, n: int = 20) -> List[dict]:
    key = chat_key(session_id)
    raw = r.lrange(key, -n, -1)
    return [json.loads(x) for x in raw]


def format_history_for_prompt(history: List[dict], max_chars: int = 1200) -> str:
    lines = []
    total = 0
    for m in history:
        line = f"{m.get('role','').upper()}: {m.get('content','')}".strip()
        if not line:
            continue
        if total + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines) if lines else "(yok)"


def memory_tail_for_cache(session_id: str, n: int = 6) -> List[str]:
    """
    ✅ Stabil: sadece USER mesajlarından tail alıyoruz.
    (assistant cevabı sürekli değiştiği için cache key'i bozmasın)
    """
    hist = get_history(session_id, n=60)
    user_msgs = [m for m in hist if m.get("role") == "user"]
    tail_msgs = user_msgs[-n:] if n > 0 else []
    tail = []
    for m in tail_msgs:
        content = (m.get("content", "") or "")[:200]
        tail.append(f"user:{content}")
    return tail


# -----------------------------
# ANSWER CACHE
# -----------------------------
def _hash_obj(obj) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def make_cache_key(payload: dict) -> str:
    return "ans:" + _hash_obj(payload)


def get_cached_answer(cache_key: str) -> Optional[dict]:
    val = r.get(cache_key)
    return json.loads(val) if val else None


def set_cached_answer(cache_key: str, answer_obj: dict) -> None:
    r.setex(cache_key, ANSWER_TTL_SEC, json.dumps(answer_obj, ensure_ascii=False))


# -----------------------------
# HELPERS
# -----------------------------
def sanitize_question(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"^\s*(>+|\*+|\-+|#+|\|+)\s*", "", text)
    text = text.replace("```", "")
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_or_load_vectorstore(pdf_path: str) -> FAISS:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.isdir(INDEX_DIR) and os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print("✅ Index bulundu. Diskten yükleniyor...")
        t0 = time.time()
        vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        print(f"✅ Yüklendi. Süre: {time.time() - t0:.2f}s")
        return vs

    print("📄 Index yok. PDF okunuyor ve index oluşturuluyor (ilk sefer uzun sürebilir)...")
    t0 = time.time()

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    vs = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)

    print(f"✅ Index oluşturuldu ve kaydedildi. Süre: {time.time() - t0:.2f}s")
    return vs


def extract_pages(source_documents) -> List[str]:
    pages = []
    for d in source_documents or []:
        p = d.metadata.get("page", None)
        if p is not None:
            pages.append(str(p))

    def key(x):
        return int(x) if x.isdigit() else 10**9

    return sorted(set(pages), key=key)


def docs_to_context(docs, max_chars: int = 8000) -> str:
    parts = []
    total = 0
    for d in docs:
        text = (d.page_content or "").strip()
        if not text:
            continue
        if total + len(text) > max_chars:
            remain = max_chars - total
            if remain > 200:
                parts.append(text[:remain])
            break
        parts.append(text)
        total += len(text)
    return "\n\n---\n\n".join(parts)


def llm_text(out) -> str:
    return getattr(out, "content", str(out)).strip()


# -----------------------------
# ✅ MEMORY ROUTER (NEW, non-ilkel)
# -----------------------------
TR_STOP = {
    "ve", "veya", "ile", "de", "da", "mi", "mı", "mu", "mü",
    "ben", "benim", "bana", "bende", "benden", "sen", "sana", "sende",
    "ne", "nedir", "neydi", "kaç", "kim", "hangi", "nasıl",
    "bir", "bu", "şu", "o", "çok", "az", "daha", "önce", "sonra"
}

MEMORY_HINT_RE = re.compile(
    r"\b(az önce|demin|daha önce|önce|konuşmada|sohbette|hatırla|hatırlıyor musun|"
    r"söylemiştim|demiştim|yazmıştım|bahsetmiştim|sen demiştin|ben demiştim)\b",
    re.IGNORECASE
)

PERSONAL_HINT_RE = re.compile(
    r"\b(benim|bana|bende|benden|adım|ismim|yaşım|okulum|üniversitem|nereliyim)\b",
    re.IGNORECASE
)


def tokenize(text: str) -> List[str]:
    text = sanitize_question(text.lower())
    text = re.sub(r"[^a-zçğıöşü0-9\s]", " ", text)
    toks = [t for t in text.split() if t and t not in TR_STOP and len(t) > 1]
    return toks


def overlap_score(q_toks: List[str], msg_toks: List[str]) -> int:
    if not q_toks or not msg_toks:
        return 0
    return len(set(q_toks) & set(msg_toks))


def looks_like_memory_question(question: str) -> bool:
    q = question.lower()
    return bool(MEMORY_HINT_RE.search(q) or PERSONAL_HINT_RE.search(q))


def try_answer_from_memory(session_id: str, question: str) -> Optional[str]:
    """
    ✅ Liste yok.
    - Önce "konuşmaya/kişisel bilgiye referans var mı" bakar
    - Varsa chat memory içinde arar
    - Bulursa RAG'e girmeden döner
    """
    if not looks_like_memory_question(question):
        return None

    hist = get_history(session_id, n=60)
    qlow = question.lower()

    # 1) İsim çıkarımı (çok güvenilir)
    if re.search(r"\b(adım|ismim)\b", qlow) and re.search(r"\b(ne|nedir|neydi)\b", qlow):
        for m in reversed(hist):
            if m.get("role") == "user":
                text = m.get("content", "")
                m2 = re.search(r"\b(ad[ıi]m|ismim)\s+([A-Za-zÇĞİÖŞÜçğıöşü]+)", text, re.IGNORECASE)
                if m2:
                    return f"Adın {m2.group(2)}."
        return "Konuşmada adını bulamadım."

    # 2) "az önce ne yazdım" → bir önceki user mesajını ver
    if re.search(r"\b(az önce|demin|son mesaj|ne yazdım|ne yazmıştım)\b", qlow):
        # Son user mesajını bul
        for m in reversed(hist):
            if m.get("role") == "user":
                return f"Az önce şunu yazdın: {m.get('content','')}"
        return "Önceki mesajını bulamadım."

    # 3) Genel: kelime örtüşmesine göre en ilgili user mesajını bul
    q_toks = tokenize(question)
    best_content = ""
    best_score = 0

    for m in reversed(hist):
        if m.get("role") != "user":
            continue
        content = m.get("content", "") or ""
        s = overlap_score(q_toks, tokenize(content))
        if s > best_score:
            best_score = s
            best_content = content

    if best_score >= 2:
        return f"Konuşmada bununla ilgili şunu yazmıştın: {best_content}"

    return None


# -----------------------------
# ✅ RETRIEVAL SPEED + REWRITE GUARD (IMPROVED)
# -----------------------------
BAD_REWRITE_HINTS = {
    "merhaba", "yardımcı", "olabilirim", "yazın", "lütfen", "ben", "sen", "size"
}

def is_bad_rewrite(text: str) -> bool:
    """
    ✅ Rewrite çıktısı "arama sorgusu" yerine sohbet cümlesine dönerse yakala.
    """
    t = sanitize_question(text.lower())
    if len(t.split()) == 0:
        return True
    # çok uzun ya da cümle gibi
    if len(t.split()) > MAX_REWRITE_WORDS:
        return True
    # sohbet kokusu
    bad_hits = sum(1 for w in t.split() if w in BAD_REWRITE_HINTS)
    if bad_hits >= 2:
        return True
    return False


def rewrite_query(llm: ChatOllama, question: str) -> str:
    """
    ✅ Guard eklendi: rewrite saçmalarsa question'a fallback.
    """
    try:
        out = llm.invoke(REWRITE_PROMPT.format(question=question))
        text = sanitize_question(llm_text(out))
        words = text.split()
        candidate = " ".join(words[:MAX_REWRITE_WORDS]) if words else question
        if is_bad_rewrite(candidate):
            return question
        return candidate
    except Exception:
        return question


def suggest_fallback_queries(llm: ChatOllama, question: str) -> List[str]:
    try:
        out = llm.invoke(FALLBACK_PROMPT.format(question=question))
        text = llm_text(out)
        lines = [sanitize_question(x) for x in text.splitlines()]
        lines = [x for x in lines if x]
        return lines[:3]
    except Exception:
        return []


def retrieve_with_fallback(retriever, llm: ChatOllama, question: str) -> Tuple[str, List]:
    """
    ✅ Hız iyileştirmesi:
    - <= 6 kelime sorularda rewrite LLM çağrısı YOK (direkt question)
    """
    if len(question.split()) <= 6:
        search_query = question
    else:
        search_query = rewrite_query(llm, question)

    docs = retriever.get_relevant_documents(search_query)

    if len(docs) >= MIN_DOCS:
        return search_query, docs

    suggestions = suggest_fallback_queries(llm, question)
    if suggestions:
        print("\n⚠️ Retrieval zayıf. 'Did you mean?' denemeleri:")
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}) {s}")

    for s in suggestions:
        docs2 = retriever.get_relevant_documents(s)
        if len(docs2) >= MIN_DOCS:
            print(f"✅ Fallback sorgu ile bulundu: {s}")
            return s, docs2

    return search_query, docs


def answer_with_llm(llm: ChatOllama, question: str, docs, chat_history_text: str) -> str:
    context = docs_to_context(docs)
    prompt = ANSWER_PROMPT.format(
        context=context,
        question=question,
        no_answer_text=NO_ANSWER_TEXT,
        chat_history=chat_history_text,
    )
    out = llm.invoke(prompt)
    return llm_text(out)


# -----------------------------
# MAIN
# -----------------------------
def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError("data/doc.pdf bulunamadı. PDF’yi data/ içine koy ve adını doc.pdf yap.")

    # Redis ping
    try:
        if not r.ping():
            raise RuntimeError("Redis ping false döndü.")
    except Exception as e:
        raise RuntimeError(
            f"Redis'e bağlanılamadı ({REDIS_HOST}:{REDIS_PORT}). Docker çalışıyor mu?\nHata: {e}"
        )

    vectorstore = build_or_load_vectorstore(PDF_PATH)

    # BM25 hazırlık
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K

    faiss_ret = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    retriever = EnsembleRetriever(
        retrievers=[bm25, faiss_ret],
        weights=[BM25_WEIGHT, FAISS_WEIGHT],
    )

    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)

    session_id = os.getenv("SESSION_ID", "local-cli")

    print(
        f"\n✅ Hazır. (chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}, k={TOP_K}, llm={LLM_MODEL}, hybrid=BM25+FAISS, single-pass)"
    )
    print(f"✅ Redis: {REDIS_HOST}:{REDIS_PORT} | session_id={session_id}")
    print("Sorunu yaz (çıkmak için 'q'):\n")

    while True:
        raw = input("> ")
        question = sanitize_question(raw)

        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue

        # ✅ 0) MEMORY ROUTER: Kullanıcı mesajını kaydetmeden önce kontrol et
        # (az önce ne yazdım gibi soruların doğru çalışması için)
        mem_answer = try_answer_from_memory(session_id, question)
        if mem_answer:
            add_message(session_id, "user", question)
            add_message(session_id, "assistant", mem_answer)

            print(f"\n--- CEVAP (memory) ---")
            print(mem_answer)
            print("\nKaynak: (chat memory)")
            print("(debug) search_query: (memory)\n")
            continue

        # ✅ (MEMORY) kullanıcı mesajını konuşmaya ekle
        add_message(session_id, "user", question)

        # 1) Retrieval
        used_search_query, retrieved_docs = retrieve_with_fallback(retriever, llm, question)

        if not retrieved_docs:
            add_message(session_id, "assistant", NO_ANSWER_TEXT)

            print(f"\n--- CEVAP (0.00s) ---")
            print(NO_ANSWER_TEXT)
            print("\nKaynak: (retrieval boş)")
            print(f"(debug) search_query: {used_search_query}\n")
            continue

        pages = extract_pages(retrieved_docs)

        # 2) Cache key
        mem_tail = memory_tail_for_cache(session_id, n=MEM_TAIL_FOR_CACHE)
        cache_payload = {
            "q": question,
            "search_query": used_search_query,
            "pages": pages,
            "llm_model": LLM_MODEL,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "prompt_sig": _hash_obj({"answer_prompt": ANSWER_PROMPT.template, "no_answer": NO_ANSWER_TEXT}),
            "mem_tail": mem_tail,
            "ver": 2,  # ✅ versiyon arttı (yeni routing/guard)
        }
        cache_key = make_cache_key(cache_payload)

        cached = get_cached_answer(cache_key)
        if cached:
            add_message(session_id, "assistant", cached.get("answer", ""))

            print(f"\n--- CEVAP (cache) ---")
            print(cached.get("answer", ""))
            if cached.get("pages"):
                print(f"\nKaynak: sayfa {', '.join(cached['pages'])}")
            else:
                print("\nKaynak: (sayfa bulunamadı)")
            print(f"(debug) search_query: {used_search_query}\n")
            continue

        # 3) Prompt history
        hist = get_history(session_id, n=MEM_TURNS_FOR_PROMPT)
        chat_history_text = format_history_for_prompt(hist)

        # 4) LLM answer
        t0 = time.time()
        answer = answer_with_llm(llm, question, retrieved_docs, chat_history_text)
        elapsed = time.time() - t0

        # 5) Cache write
        cache_obj = {
            "answer": answer,
            "pages": pages,
            "cached": False,
            "ts": int(time.time()),
        }
        set_cached_answer(cache_key, cache_obj)

        # 6) Memory write
        add_message(session_id, "assistant", answer)

        print(f"\n--- CEVAP ({elapsed:.2f}s) ---")
        print(answer)

        if pages:
            print(f"\nKaynak: sayfa {', '.join(pages)}")
        else:
            print("\nKaynak: (sayfa bulunamadı)")

        print(f"(debug) search_query: {used_search_query}\n")


if __name__ == "__main__":
    main()
