import io
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pdf2image import convert_from_path, pdfinfo_from_path

# ================================
# CONFIG
# ================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "ocr_input"
OUTPUT_DIR = BASE_DIR / "ocr_output"
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_DIR = OUTPUT_DIR / "raw"
CORR_DIR = OUTPUT_DIR / "corrected"
RAW_DIR.mkdir(exist_ok=True)
CORR_DIR.mkdir(exist_ok=True)

# Load .env from multiple likely places
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.0-flash"
DPI = 300
OCR_WORKERS = int(os.getenv("OCR_WORKERS", "4"))  # Parallel OCR threads

# ================================
# PROMPTS
# ================================

SYSTEM_PROMPT_OCR = """
You are an OCR engine for Sanskrit (Devanagari), Kannada, and English.

Rules:
- Extract EXACT text as printed.
- Preserve all conjuncts (क्ष त्र ज्ञ द्य ಕ್ಶ ದ್ಯ etc.)
- Preserve anusvāra, visarga, chandrabindu, avagraha (ऽ).
- Preserve all line breaks.
- Do NOT correct anything.
- Do NOT normalize.
- Output ONLY raw Unicode text.
"""

OCR_FEWSHOT = """
### OCR Examples

Input: "विद्या विनयेन शोभते"
Output:
विद्या विनयेन शोभते

Input: "सर्वधर्मान्परित्यज्य"
Output:
सर्वधर्मान्परित्यज्य

---

Extract EXACT printed text:
"""

SYSTEM_PROMPT_CORRECT = """
You are a Sanskrit OCR-correction engine specialized in Devanagari script.

Your task is:
→ Correct ONLY OCR errors.
→ Preserve ALL original meaning, grammar, meter, structure, and line breaks.
→ Never introduce or hallucinate new content.

The output MUST strictly follow these rules:

1. **Preserve Structure**
   - Do NOT add new lines.
   - Do NOT merge or split lines.
   - Preserve punctuation, danda marks (।, ॥), and spacing exactly unless clearly an OCR error.
   - If unsure, keep the original text.

2. **Correct Only Script-Level OCR Errors**
   Fix:
   - Broken or missing **saṃyuktākṣaras** (conjunct consonants)
     (e.g., क्ष, त्र, ज्ञ, श्र, द्य, स्त्र, ह्न, स्म, ढ्य, क्त, च्छ, न्त्र, द्ग, द्ग्ध, द्ध, ख्य, ग्र, क्ल)
   - Incorrect halant (्) placement, missing virāma, or inserted virāma
   - Misrecognized characters (द/ध/ड, ग/घ, त/ऽ, ण/न, श/स/ष)
   - र्-repha mistakes (र् + consonant)
   - Long/short vowel confusion (अ/आ, इ/ई, उ/ऊ, ऋ/ॠ)
   - Broken matras (ि, ी, ु, ू, ृ, ॄ)
   - Missing **anusvāra** (ं) and **visarga** (ः)
   - Mistaken half-forms (e.g., न् + य → न्य misread as न य)
   - Confused similar glyphs (ा vs ि positioning, ं vs ँ, ष vs श)
   - Missing chandrabindu (ँ)
   - OCR artifacts (duplicate letters, unintended punctuation)

3. **Word-Level Fixes (NO translation or interpretation)**
   - Restore valid Sanskrit morphological forms when OCR breaks them.
   - Fix sandhi **only when the scanned text clearly shows a broken character**.
   - DO NOT perform semantic corrections.
   - DO NOT modernize spelling.

4. **Strict “No Hallucination” Policy**
   - If a character is ambiguous, leave it unchanged.
   - Do NOT substitute a different valid Sanskrit word unless the OCR error is obvious.
   - Never add missing portions of a verse or shloka.

5. **Goal**
   - Produce the *cleanest possible Unicode Sanskrit text* matching the original printed source.

If the input has no recognizable Sanskrit, return it unchanged.
"""


CORRECT_FEWSHOT = """
### MADHWA SCRIPTURES FEWSHOT (25 EXAMPLES)

Example 1: (Pramāṇa-lakṣaṇa)
Raw: ततोऽर्वाक्क्रमेण हसितम्
Corrected: ततोऽर्वाक्क्रमेण ह्रसितम्

Example 2: (Tattva-saṅkhyāna)
Raw: हरिर्हि परमं तत्त्व म्
Corrected: हरिर्हि परमं तत्त्वम्

Example 3: (Brahma-Sūtra-Bhāṣya)
Raw: आनन्दस्य मूल म् ब्रह्म
Corrected: आनन्दस्य मूलं ब्रह्म

Example 4: (Anuvyākhyāna)
Raw: भेद श्रुतयः न प्रकाशन्ते
Corrected: भेदश्रुतयः न प्रकाशन्ते

Example 5: (Mahābhārata-Tātparya-Nirṇaya)
Raw: भीमेन रक्षितो धर्मः
Corrected: भीमेन रक्षितो धर्मः

Example 6: (Gītā-Tātparya)
Raw: मच्चित्तः सर्व दुर्ग णैः
Corrected: मच्चित्तः सर्वदुर्गणैः

Example 7: (Harikathamṛtasāra)
Raw: हरिस्मरणे यः स्थितः
Corrected: हरिस्मरणे यः स्थितः

Example 8: (Nyāyasudhā ref)
Raw: शब्द प्रमाण मित्युक्तम्
Corrected: शब्दप्रमाणमित्युक्तम्

Example 9: (Bhāgavata-Tātparya)
Raw: भक्त्या लभ्यते गोविन्दः
Corrected: भक्त्या लभ्यते गोविन्दः

Example 10: (Vāyu-Stuti)
Raw: नमो नमो वायू सप्तितमाय
Corrected: नमो नमो वायोऽसप्तितमाय

Example 11: (Dvadasha Stotra)
Raw: त्वं पिता च नः विश्वस्य
Corrected: त्वं पिता च नः विश्वस्य

Example 12: (Gītā-Bhāṣya)
Raw: कर्मण्येव अधिकार स्ते
Corrected: कर्मण्येव अधिकारस्ते

Example 13: (Mahābhārata-Tātparya)
Raw: भीमः एक एव असुरान् हन्ति
Corrected: भीम एक एव असुरान् हन्ति

Example 14: (Pramāṇa-lakṣaṇa)
Raw: निर्दोषोप पत्तिर् अनुमान म्
Corrected: निर्दोषोपपत्तिरनुमानम्

Example 15: (Tattva-saṅkhyāna)
Raw: जीवोऽणुः स्यात् परतन्त्रः
Corrected: जीवोऽणुः स्यात् परतन्त्रः

Example 16: (Anuvyākhyāna)
Raw: विष्णोः सर्वोत्कर्ष निर्देशकाः
Corrected: विष्णोः सर्वोत्कर्षनिर्देशकाः

Example 17: (Nyāyasudhā)
Raw: साधन चतु ष्टयं निर्दिष्ट म्
Corrected: साधनचतुष्टयं निर्दिष्टम्

Example 18: (Bhāgavata-Tātparya)
Raw: कालः स्वभाव त् एव कार्यम्
Corrected: कालः स्वभावादेव कार्यम्

Example 19: (Gītā-Tātparya)
Raw: भक्तोऽहं न प्रिया हि कर्म णि
Corrected: भक्तोऽहं न प्रिया हि कर्मणि

Example 20: (Harikathamṛtasāra)
Raw: हरि भक्तिर्वर्धते साधुसङ्गेन
Corrected: हरिभक्तिर्वर्धते साधुसङ्गेन

Example 21: (Pramāṇa-lakṣaṇa)
Raw: अर्थापत्ति उपमे अनुमाविशेषः
Corrected: अर्थापत्त्युपमे अनुमाविशेषः

Example 22: (Mahābhārata-Tatparya)
Raw: दुर्योध नस्य चित्त दुष्टता
Corrected: दुर्योधनस्य चित्तदुष्टता

Example 23: (Brahma-Sūtra-Bhāṣya)
Raw: भेद एव नित्य सिद्धः
Corrected: भेद एव नित्यसिद्धः

Example 24: (Anuvyākhyāna)
Raw: तत्र परमात्मा निर दोषः
Corrected: तत्र परमात्मा निर्दोषः

Example 25: (Harikathamṛtasāra)
Raw: श्रीकृष्णार्पण मस्तु अन्ते
Corrected: श्रीकृष्णार्पणमस्त्वन्ते


Correct the following text:
"""

# ================================
# GEMINI MODEL
# ================================

def build_model(system_prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing GEMINI_API_KEY in .env")

    genai.configure(api_key=api_key)

    safety = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
        safety_settings=safety,
        generation_config={
            "temperature": 0,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
    )
    return model


# ================================
# PAGE SPLITTER (TALL SANSKRIT)
# ================================
def split_vertical(page):
    w, h = page.size
    if h > w * 1.4:
        mid = h // 2
        return [page.crop((0, 0, w, mid)), page.crop((0, mid, w, h))]
    return [page]


# ================================
# PASS 1 — OCR
# ================================
def ocr_page(model, img, max_retries=3):
    chunks = split_vertical(img)
    out = []

    for c in chunks:
        buf = io.BytesIO()
        c.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # Retry with exponential backoff
        for attempt in range(max_retries):
            try:
                r = model.generate_content(
                    [OCR_FEWSHOT, {"mime_type": "image/png", "data": img_bytes}],
                    request_options={"timeout": 300},  # Increased to 5 minutes
                )

                # Check if response has candidates
                if not r.candidates or len(r.candidates) == 0:
                    print(f" ⚠️  Empty response from API (no candidates), retrying...")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f" ⚠️  Skipping chunk after {max_retries} empty responses")
                        out.append("")  # Add empty text for this chunk
                        break

                text = "".join(
                    (p.text or "") for p in r.candidates[0].content.parts
                ).strip()

                out.append(text)
                break  # Success, exit retry loop

            except IndexError as e:
                # Handle empty candidates list
                print(f" ⚠️  Empty response (IndexError), retrying...")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    time.sleep(wait_time)
                else:
                    print(f" ⚠️  Skipping chunk after {max_retries} attempts")
                    out.append("")  # Add empty text for this chunk
                    break

            except Exception as e:
                if "DeadlineExceeded" in str(e) or "504" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                        print(f" ⚠️  Timeout, retrying in {wait_time}s... (attempt {attempt + 2}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f" ❌ Failed after {max_retries} attempts")
                        raise
                else:
                    raise  # Re-raise non-timeout errors immediately

    return "\n".join(out)


# ================================
# PASS 2 — CORRECTION
# ================================
def correct_text(model, raw, max_retries=3):
    # Retry with exponential backoff
    for attempt in range(max_retries):
        try:
            r = model.generate_content(
                CORRECT_FEWSHOT + raw,
                request_options={"timeout": 300}  # Increased to 5 minutes
            )

            # Check if response has candidates
            if not r.candidates or len(r.candidates) == 0:
                print(f" ⚠️  Empty response from correction API, retrying...")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    time.sleep(wait_time)
                    continue
                else:
                    print(f" ⚠️  Returning raw text after {max_retries} empty responses")
                    return raw  # Return uncorrected text

            return r.text.strip()

        except (IndexError, AttributeError) as e:
            # Handle empty candidates or missing text attribute
            print(f" ⚠️  Empty response from correction API, retrying...")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                time.sleep(wait_time)
            else:
                print(f" ⚠️  Returning raw text after {max_retries} attempts")
                return raw  # Return uncorrected text

        except Exception as e:
            if "DeadlineExceeded" in str(e) or "504" in str(e):
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                    print(f" ⚠️  Correction timeout, retrying in {wait_time}s... (attempt {attempt + 2}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f" ❌ Correction failed after {max_retries} attempts")
                    raise
            else:
                raise  # Re-raise non-timeout errors immediately


# ================================
# SELECT PDF AUTO-DETECT
# ================================
def select_pdf():
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print("❌ No PDF files in ocr_input/")
        return None

    print("\nAvailable PDFs in ./ocr_input:\n")
    for i, p in enumerate(pdfs, 1):
        print(f"{i}. {p.name}")

    idx = int(input(f"Select (1-{len(pdfs)}): "))
    return pdfs[idx - 1]


# ================================
# INTERACTIVE INPUT HELPERS
# ================================
def get_int_input(prompt: str, min_val: int = 1) -> int:
    """Get integer input with validation"""
    while True:
        try:
            value = input(prompt).strip()
            if not value:
                continue
            num = int(value)
            if num >= min_val:
                return num
            print(f"❌ Please enter a number >= {min_val}")
        except ValueError:
            print("❌ Please enter a valid number")


def get_start_page() -> int:
    """Interactive start page input"""
    print("\n" + "="*70)
    print("📄 START PAGE CONFIGURATION")
    print("="*70)
    print("""
This tells the OCR where printed "Page 1" begins in the PDF.

Examples:
  - If the PDF has 5 pages of front matter, then printed Page 1 is at OCR page 6
  - If printed Page 1 is the first page of the PDF, enter 1
  - Front matter pages will be labeled as "[Front Matter p.1]", "[Front Matter p.2]", etc.
""")

    start_page = get_int_input("📍 Enter OCR page number where printed Page 1 begins: ", min_val=1)

    print(f"\n✅ Start page set to: {start_page}")
    if start_page > 1:
        print(f"   Pages 1-{start_page-1} will be labeled as Front Matter")

    return start_page


def get_ocr_mode() -> str:
    """Ask user whether to do one-pass or two-pass OCR"""
    print("\n" + "="*70)
    print("🔧 OCR MODE SELECTION")
    print("="*70)
    print("""
Choose OCR processing mode:

1. ONE-PASS (Fast)
   - Only performs OCR extraction
   - No correction/cleanup
   - Faster processing
   - Use for: Quick previews, testing

2. TWO-PASS (Accurate)
   - Pass 1: OCR extraction
   - Pass 2: AI-based correction (fixes conjuncts, sandhi, OCR errors)
   - Slower but more accurate
   - Use for: Final production, ingestion into database
""")

    while True:
        choice = input("📍 Select mode (1 for one-pass, 2 for two-pass): ").strip()
        if choice == "1":
            print("\n✅ Mode: ONE-PASS (OCR only)")
            return "one-pass"
        elif choice == "2":
            print("\n✅ Mode: TWO-PASS (OCR + Correction)")
            return "two-pass"
        else:
            print("❌ Please enter 1 or 2")


def get_batch_size() -> int:
    """Ask user for batch size to control memory usage"""
    print("\n" + "="*70)
    print("💾 MEMORY CONFIGURATION")
    print("="*70)
    print("""
For large PDFs, processing pages in batches prevents out-of-memory errors.

Batch size recommendations:
  - Small PDFs (<100 pages):     20-50 pages per batch
  - Medium PDFs (100-500 pages): 10-20 pages per batch
  - Large PDFs (500+ pages):     5-10 pages per batch
  - Huge PDFs (1000+ pages):     2-5 pages per batch

Smaller batches = Lower memory usage but slightly slower
Larger batches = Faster but may cause memory errors on huge files
""")

    while True:
        choice = input("📍 Enter batch size (press Enter for default 10): ").strip()
        if not choice:
            print("\n✅ Using default batch size: 10 pages")
            return 10
        try:
            batch_size = int(choice)
            if batch_size >= 1:
                print(f"\n✅ Batch size set to: {batch_size} pages")
                return batch_size
            else:
                print("❌ Batch size must be at least 1")
        except ValueError:
            print("❌ Please enter a valid number")


# ================================
# FULL OCR PIPELINE (MEMORY-EFFICIENT)
# ================================
def run_pipeline(pdf_path, start_page=1, mode="two-pass", batch_size=10):
    """
    Memory-efficient OCR pipeline that processes pages in batches.

    Args:
        pdf_path: Path to PDF file
        start_page: OCR page number where printed Page 1 begins
        mode: "one-pass" or "two-pass"
        batch_size: Number of pages to process at once (reduces memory usage)
    """
    print("📄 Analyzing PDF…")

    # Get total page count WITHOUT loading all pages into memory
    info = pdfinfo_from_path(pdf_path)
    total_pages = info.get("Pages", 0)
    print(f"✓ Found {total_pages} pages")

    if total_pages == 0:
        print("❌ Could not determine page count")
        return None

    # Build models based on mode
    ocr_model = build_model(SYSTEM_PROMPT_OCR)
    if mode == "two-pass":
        corr_model = build_model(SYSTEM_PROMPT_CORRECT)

    raw_pages = []
    corr_pages = []

    def _ocr_task(idx, page, total):
        printed_page = idx - start_page + 1
        page_label = f"Page {printed_page}" if printed_page >= 1 else f"[Front Matter p.{idx}]"
        print(f"OCR {page_label} (OCR page {idx}/{total}) …")
        raw = ocr_page(ocr_model, page)
        return idx, page_label, raw

    # Process pages in batches to avoid OOM
    print(f"\n🔄 Processing in batches of {batch_size} pages to conserve memory…\n")

    for batch_start in range(1, total_pages + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, total_pages)
        print(f"\n📦 BATCH: Pages {batch_start}-{batch_end} ({batch_end - batch_start + 1} pages)")

        # Load ONLY this batch into memory
        batch_pages = convert_from_path(
            pdf_path,
            dpi=DPI,
            first_page=batch_start,
            last_page=batch_end
        )

        # Pass 1: OCR in parallel for this batch
        with ThreadPoolExecutor(max_workers=OCR_WORKERS) as executor:
            futures = [
                executor.submit(_ocr_task, batch_start + i, page, total_pages)
                for i, page in enumerate(batch_pages)
            ]
            batch_results = []
            for fut in as_completed(futures):
                batch_results.append(fut.result())

        # Preserve original order within batch
        batch_results.sort(key=lambda x: x[0])

        # Pass 2: Correction (only if two-pass mode)
        for idx, page_label, raw in batch_results:
            if mode == "two-pass":
                print(f"Correcting {page_label} …")
                corrected = correct_text(corr_model, raw)
            else:
                corrected = None

            raw_with_header = f"\n{'='*20} {page_label} {'='*20}\n{raw}"
            raw_pages.append(raw_with_header)

            if mode == "two-pass":
                corr_with_header = f"\n{'='*20} {page_label} {'='*20}\n{corrected}"
                corr_pages.append(corr_with_header)

        # Clear batch from memory
        del batch_pages
        del batch_results
        import gc
        gc.collect()

        print(f"✓ Batch {batch_start}-{batch_end} completed")

    # Full concatenated output
    full_raw = "\n\n".join(raw_pages)

    # Save raw OCR
    raw_path = RAW_DIR / f"{pdf_path.stem}_ocr_raw.txt"
    raw_path.write_text(full_raw, encoding="utf-8")
    print("\n📌 RAW OCR Saved:", raw_path)

    # Save corrected (only if two-pass)
    if mode == "two-pass":
        full_corr = "\n\n".join(corr_pages)
        corr_path = CORR_DIR / f"{pdf_path.stem}_ocr_corrected.txt"
        corr_path.write_text(full_corr, encoding="utf-8")
        print("📌 Corrected OCR Saved:", corr_path)

    # Preview output
    print("\n====== RAW OCR (First 2000 chars) ======\n")
    print(full_raw[:2000], "\n...")

    if mode == "two-pass":
        print("\n====== CORRECTED OCR (First 2000 chars) ======\n")
        print(full_corr[:2000], "\n...")
        return full_raw, full_corr
    else:
        return full_raw


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    print("\n======== MADHWA OCR EXTRACTION ========\n")

    pdf = select_pdf()
    if not pdf:
        sys.exit()

    # Get OCR mode (one-pass or two-pass)
    mode = get_ocr_mode()

    # Get batch size for memory-efficient processing
    batch_size = get_batch_size()

    # Get start page for proper page numbering
    start_page = get_start_page()

    # Run OCR pipeline with selected options
    run_pipeline(pdf, start_page=start_page, mode=mode, batch_size=batch_size)
