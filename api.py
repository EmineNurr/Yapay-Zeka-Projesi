# app/api.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.predict import VerifiedPurchasePredictor
from src.config import CFG

app = FastAPI(title="VerifiedPurchase Detector API", version="1.0.0")

# Uygulama açılınca modeli 1 kez yükle
predictor = VerifiedPurchasePredictor(threshold=CFG.DEFAULT_THRESHOLD)


class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_api(req: PredictRequest):
    pred, proba_true = predictor.predict(req.text)
    return {
        "verifiedPurchase_pred": bool(pred),
        "verifiedPurchase_proba_true": float(proba_true),
        "threshold": predictor.threshold
    }


def render_page(text_value: str = "", result: dict | None = None, error: str | None = None) -> str:
    badge_html = ""
    meter_width = 0.0
    proba_pct = None
    desc = ""

    if result is not None:
        proba_pct = round(result["verifiedPurchase_proba_true"] * 100, 2)
        meter_width = max(0.0, min(100.0, proba_pct))

        if result["verifiedPurchase_pred"] is True:
            badge_html = '<span class="badge badge-green">Verified Purchase</span>'
            desc = "Bu yorumun, ürünü platform üzerinden satın almış bir kullanıcı tarafından yazılmış olma ihtimali daha yüksektir."
        else:
            badge_html = '<span class="badge badge-red">Non-Verified</span>'
            desc = "Bu yorumun, ürünü platform üzerinden satın almadan yazılmış olma ihtimali daha yüksektir."

    json_block = ""
    if result is not None:
        json_block = f"""
        <details class="details">
          <summary>Teknik detayları göster (JSON)</summary>
          <pre class="json-box">{result}</pre>
        </details>
        """

    error_block = ""
    if error:
        error_block = f"""
        <div class="alert">
          <b>Hata:</b> {error}
        </div>
        """

    result_block = ""
    if result is not None:
        result_block = f"""
        <div class="result-card">
          <div class="result-header">
            {badge_html}
            <span class="confidence-text">
              Verified olma olasılığı: <b>{proba_pct}%</b>
              <span class="tiny"> (threshold: {result.get("threshold", predictor.threshold)})</span>
            </span>
          </div>

          <div class="meter">
            <div class="meter-bar" style="width: {meter_width}%;"></div>
          </div>

          <div class="result-desc">
            {desc}
            <div class="hint">
              Not: Model yalnızca yorum metninden tahmin üretir. Kesin kanıt değildir.
            </div>
          </div>

          {json_block}
        </div>
        """

    return f"""
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Verified Purchase Detector</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f5f7fb;
      color: #0f172a;
    }}
    .container {{
      max-width: 980px;
      margin: 40px auto;
      padding: 0 18px;
    }}
    .header {{ margin-bottom: 18px; }}
    .title {{ font-size: 40px; margin: 0; font-weight: 800; }}
    .subtitle {{
      margin-top: 10px;
      color: #475569;
      font-size: 16px;
      line-height: 1.4;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #e8e8ef;
      border-radius: 18px;
      padding: 22px;
      box-shadow: 0 10px 30px rgba(20, 20, 60, 0.06);
    }}
    label {{
      font-weight: 700;
      font-size: 18px;
      display: block;
      margin-bottom: 10px;
    }}
    textarea {{
      width: 100%;
      height: 150px;
      resize: vertical;
      padding: 14px;
      border-radius: 14px;
      border: 1px solid #dbe2f0;
      font-size: 15px;
      outline: none;
    }}
    textarea:focus {{
      border-color: #4f46e5;
      box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.10);
    }}
    .actions {{
      margin-top: 14px;
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    .btn {{
      border: none;
      background: #4f46e5;
      color: white;
      font-weight: 700;
      padding: 12px 18px;
      border-radius: 12px;
      cursor: pointer;
      font-size: 15px;
    }}
    .btn:hover {{ filter: brightness(0.95); }}
    .muted {{
      margin-top: 10px;
      color: #64748b;
      font-size: 13px;
    }}
    .alert {{
      margin-top: 14px;
      padding: 12px 14px;
      border-radius: 14px;
      background: #fff1f2;
      border: 1px solid #fecdd3;
      color: #9f1239;
    }}
    .result-card {{
      margin-top: 18px;
      background: #ffffff;
      border: 1px solid #e8e8ef;
      border-radius: 18px;
      padding: 18px 18px;
      box-shadow: 0 10px 30px rgba(20, 20, 60, 0.06);
    }}
    .result-header {{
      display:flex;
      align-items:center;
      gap:12px;
      flex-wrap:wrap;
    }}
    .badge {{
      display:inline-flex;
      align-items:center;
      padding: 8px 12px;
      border-radius: 999px;
      font-weight: 800;
      font-size: 13px;
      letter-spacing: .2px;
    }}
    .badge-green {{ background:#e9fbef; color:#137a35; border:1px solid #bff0cd; }}
    .badge-red {{ background:#ffecec; color:#a21717; border:1px solid #ffc7c7; }}
    .confidence-text {{ color:#1f2937; font-size: 15px; }}
    .tiny {{ color:#64748b; font-size: 12px; margin-left: 6px; }}
    .meter {{
      margin-top: 12px;
      height: 12px;
      border-radius: 999px;
      background: #f1f3f8;
      overflow:hidden;
      border:1px solid #e6e8f2;
    }}
    .meter-bar {{
      height:100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #4f46e5, #22c55e);
    }}
    .result-desc {{ margin-top: 12px; color:#111827; line-height: 1.5; }}
    .hint {{ margin-top: 8px; color:#6b7280; font-size: 13px; }}
    .details {{ margin-top: 14px; }}
    .details summary {{
      cursor:pointer;
      color:#374151;
      font-weight: 700;
      font-size: 14px;
    }}
    .json-box {{
      margin-top: 10px;
      background:#0b1220;
      color:#e5e7eb;
      padding: 12px;
      border-radius: 12px;
      overflow:auto;
      font-size: 12.5px;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1 class="title">Verified Purchase Detector</h1>
      <div class="subtitle">
        E-ticaret yorumlarından kullanıcının ürünü platform üzerinden satın alıp almadığını tahmin eden yapay zekâ demosu.
      </div>
    </div>

    <div class="card">
      <form method="POST" action="/ui">
        <label for="text">Yorum Metni</label>
        <textarea id="text" name="text" placeholder="Örn: Kargo çok geç geldi, memnun kalmadım...">{text_value}</textarea>
        <div class="actions">
          <button class="btn" type="submit">Tahmin Et</button>
        </div>
        <div class="muted">
          Yorumunuzu <b>Türkçe</b> yazabilirsiniz. Sistem arka planda İngilizce’ye çevirerek analiz eder.
        </div>
      </form>
      {error_block}
    </div>

    {result_block}
  </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return render_page()


@app.get("/ui", response_class=HTMLResponse)
def ui_get():
    return render_page()


@app.post("/ui", response_class=HTMLResponse)
async def ui_post(request: Request):
    form = await request.form()
    text = (form.get("text") or "").strip()

    if not text:
        return render_page(text_value="", result=None, error="Lütfen yorum metni gir.")

    try:
        pred, proba_true = predictor.predict(text)
        result = {
            "verifiedPurchase_pred": bool(pred),
            "verifiedPurchase_proba_true": float(proba_true),
            "threshold": predictor.threshold
        }
        return render_page(text_value=text, result=result, error=None)
    except Exception as e:
        return render_page(text_value=text, result=None, error=str(e))
