import os, torch, librosa, uvicorn, re, json
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
from difflib import SequenceMatcher

# 🌟 API കീ പരിസ്ഥിതി വേരിയബിളിൽ നിന്ന് എടുക്കുന്നു (GitHub/HF Secrets)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("❌ GEMINI_API_KEY environment variable is not set!")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI()

# മോഡൽ ലോഡ് ചെയ്യുന്നു
MODEL_ID = "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v3"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

def clean_arabic(text):
    text = re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]', '', text)
    text = re.sub(r'[إأآءٱ]', 'ا', text)
    return text.replace('ة', 'ه').replace('ى', 'ي').strip()

def get_word_timestamps(logits, audio_duration_ms):
    predicted_ids = torch.argmax(logits, dim=-1)[0]
    tokens = processor.tokenizer.convert_ids_to_tokens(predicted_ids)
    words, current_word, start_frame =[], "", 0
    total_frames = len(predicted_ids)
    for i, token in enumerate(tokens):
        if token == "|":
            if current_word:
                words.append({"word": current_word, "start": (start_frame / total_frames) * audio_duration_ms, "end": (i / total_frames) * audio_duration_ms})
                current_word = ""
            start_frame = i + 1
        elif token not in [processor.tokenizer.pad_token, processor.tokenizer.word_delimiter_token]:
            current_word += token
    return words

def get_word_alignment_analysis(s_words, e_words):
    matcher = SequenceMatcher(None, [clean_arabic(w) for w in e_words], [clean_arabic(w) for w in s_words])
    result_map = {i: {"status": 2, "heard": "വിട്ടുപോയി"} for i in range(len(e_words))}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for idx, i in enumerate(range(i1, i2)):
                spoken_idx = j1 + idx
                if spoken_idx < len(s_words):
                    result_map[i] = {"status": 1, "heard": s_words[spoken_idx]}
        elif tag in ['replace', 'delete', 'insert']:
            for idx, i in enumerate(range(i1, i2)):
                spoken_idx = j1 + idx
                if spoken_idx < len(s_words):
                    result_map[i] = {"status": 2, "heard": s_words[spoken_idx]}
    return result_map

async def verify_with_gemini(audio_path, expected_word, is_sabaq=False, full_text="", language="Malayalam"):
    try:
        sample_file = genai.upload_file(path=audio_path)
        if is_sabaq:
            prompt = f"Listen to this recitation and compare with: '{full_text}'. Return ONLY valid JSON with 'overall_accuracy' and 'mistakes_list' (surah_id, ayah_number, word, expected, explanation, start_time, end_time)."
        else:
            prompt = f"Is the word '{expected_word}' pronounced correctly? Reply EXACTLY: Verdict: [True/False]\nExplanation: [Malayalam explanation]\nStart: [sec]\nEnd: [sec]"
        
        response = gemini_model.generate_content([prompt, sample_file])
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

@app.post("/openai/v1/audio/transcriptions")
async def dawra_transcribe(file: UploadFile = File(...), expected_text: str = Form("")):
    temp_path = "dawra_temp.m4a"
    with open(temp_path, "wb") as f: f.write(await file.read())
    try:
        audio = AudioSegment.from_file(temp_path)
        speech, _ = librosa.load(temp_path, sr=16000)
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        predicted_text = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
        timestamps = get_word_timestamps(logits, len(audio))
        
        e_words = expected_text.split()
        s_words = predicted_text.split()
        alignment_map = get_word_alignment_analysis(s_words, e_words)

        final_analysis =[]
        for i, word in enumerate(e_words):
            status = alignment_map[i]["status"]
            heard_word = alignment_map[i]["heard"]
            explanation = "ഉച്ചാരണം ശരിയാണ്." if status == 1 else f"'{heard_word}' എന്ന് കേട്ടു."
            word_start, word_end = 0.0, 0.0
            
            if status == 2 and heard_word != "വിട്ടുപോയി": 
                for ts in timestamps:
                    if clean_arabic(ts['word']) in clean_arabic(word) or clean_arabic(word) in clean_arabic(ts['word']):
                        word_start, word_end = round(ts['start']/1000, 2), round(ts['end']/1000, 2)
                        seg_path = f"seg_{i}.wav"
                        audio[max(0, ts['start']-1000):min(len(audio), ts['end']+1000)].export(seg_path, format="wav")
                        gemini_res = await verify_with_gemini(seg_path, word)
                        if "True" in gemini_res: status = 1; explanation = "ഉച്ചാരണം ശരിയാണ്."
                        else: explanation = gemini_res.split("Explanation:")[-1].strip()
                        if os.path.exists(seg_path): os.remove(seg_path)
                        break
            final_analysis.append({"expected_word": word, "heard_word": heard_word, "status": status, "explanation": explanation, "start_time": word_start, "end_time": word_end})
        
        return {"text": predicted_text, "word_analysis": final_analysis}
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.post("/analyze-sabaq")
async def sabaq_transcribe(file: UploadFile = File(...), expected_text: str = Form(""), language: str = Form("Malayalam")):
    # (Sabaq logic here...)
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
