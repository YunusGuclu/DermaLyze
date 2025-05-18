import io
import traceback
import os
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import tensorflow as tf

# Flask uygulamasını, proje kökünüzü de static dosya klasörü olarak gösterecek şekilde başlatıyoruz
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')

# --- 1) İlk model (skin cancer vs benign vs melanoma vb.) ---
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
inp  = interpreter.get_input_details()[0]
outp = interpreter.get_output_details()[0]
_, h, w, _ = inp['shape']

labels_tr = [
    'Melanositik Nevüs',
    'Melanom',
    'Benign Keratoz Benzeri Lezyon',
    'Bazal Hücre Karsinomu',
    'Aktinik Keratoz / İntraepitelyal Karsinom'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "no image"}), 400

        file = request.files['image'].read()
        img = Image.open(io.BytesIO(file)).convert('RGB')
        arr = np.array(img.resize((w, h)), dtype=np.float32)
        arr = (arr / 127.5) - 1.0
        arr = np.expand_dims(arr, axis=0)

        interpreter.set_tensor(inp['index'], arr)
        interpreter.invoke()
        probs = interpreter.get_tensor(outp['index'])[0]

        results = sorted(
            [{"label": l, "score": float(s)} for l, s in zip(labels_tr, probs)],
            key=lambda x: x["score"], reverse=True
        )
        return jsonify(results)
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


# --- 2) İkinci model (Acne.tflite) ---
acne_interpreter = tf.lite.Interpreter(model_path="Acne.tflite")
acne_interpreter.allocate_tensors()
acne_inp  = acne_interpreter.get_input_details()[0]
acne_outp = acne_interpreter.get_output_details()[0]
_, acne_h, acne_w, _ = acne_inp['shape']

label_map = {
    0: "Acne", 1: "Acne Infantile", 2: "Acne Open Comedo", 3: "Acne Steroid",
    4: "Acne conglobata", 5: "Acne excoriée", 6: "Acne keloidalis nuchae",
    7: "Acne vulgaris", 8: "Hidradenitis suppurativa", 9: "Hyperhidrosis",
    10: "Milia", 11: "Minocycline Pigmentation", 12: "Perioral Dermatitis",
    13: "Pomade acne", 14: "Rosacea"
}

disease_info = {
    "Acne": (
        "Teşhis: Deri yağ bezlerinin fazla sebum üretimi ve gözeneklerin tıkanması sonucu "
        "siyah nokta, beyaz nokta, papül ve püstüller oluşur.\n",
        "Bilgi: Ergenlik döneminde hormon dalgalanmalarıyla tetiklenir; genetik yatkınlık, stres "
        "ve beslenme de rol oynar. Topikal retinoidler, antibakteriyel losyonlar ve cilt temizliği "
        "tedavinin temelini oluşturur."
    ),
    "Acne Infantile": (
        "Teşhis: Yenidoğan ve bebeklik döneminde görülen papül-püstüller, nadiren nodül ve kistler oluşturabilir.\n",
        "Bilgi: Genellikle bir-üç aylıkken başlar, çoğu bebekte 6 aya kadar kendiliğinden geriler. "
        "Bol temiz hava, nazik cilt bakımı ve gerekirse dermatolog kontrolü önerilir."
    ),
    "Acne Open Comedo": (
        "Teşhis: Gözenek tıkanıklığındaki oksidasyonla kararan sebum tıkaçları (siyah noktalar).\n",
        "Bilgi: Hafif akne formudur, düzenli exfoliasyon ve salisilik asit içeren ürünlerle önlenebilir. "
        "Derin lezyonlara ilerlemeden müdahale edilmesi önerilir."
    ),
    "Acne Steroid": (
        "Teşhis: Topikal veya sistemik kortikosteroid kullanımına bağlı, nodüler ve kistik lezyonlar.\n",
        "Bilgi: Steroid dozu azaltılmalı, hormonal düzeyler izlenmeli, alternatif anti-enflamatuar tedaviler değerlendirilmeli."
    ),
    "Acne conglobata": (
        "Teşhis: Derin skarlı, fistüllü nodüllerle seyreden, en ağır akne tiplerinden biri.\n",
        "Bilgi: Genç erişkin erkeklerde yaygın; sistemik retinoidler ve antibiyotik kombinasyonları "
        "sıklıkla kullanılır. Erken ve agresif tedavi skarları azaltır."
    ),
    "Acne excoriée": (
        "Teşhis: Takıntılı kaşıma ve sıkma nedeniyle erozyon, skar ve pigment değişiklikleri.\n",
        "Bilgi: Psikolojik faktörler (anksiyete, obsesif davranış) ön planda; dermatolog ve psikolog "
        "iş birliğiyle hem topikal hem davranışsal müdahale gerekir."
    ),
    "Acne keloidalis nuchae": (
        "Teşhis: Ense kökünde sert, keloid benzeri nodüller ve skar dokusu.\n",
        "Bilgi: Sıkı saç kesimi, dar giyim ve travma tetikleyebilir. Siyah ırkta daha sık görülür. "
        "Cerrahi eksizyon ve intralezyonal kortikosteroid enjeksiyonları tedavide kullanılır."
    ),
    "Acne vulgaris": (
        "Teşhis: Papül, püstül, kistik lezyonlar ve komedonların karışık formu.\n",
        "Bilgi: Ergenlik döneminin tipik hastalığıdır; cilt bariyerini koruyan nazik temizleyiciler, "
        "topikal retinoidler ve antibiyotikler kombinasyonuyle kontrol altına alınır."
    ),
    "Hidradenitis suppurativa": (
        "Teşhis: Aksilla, kasık ve gluteal bölgede tekrar eden apse ve fistülsel lezyonlar.\n",
        "Bilgi: Kronik ilerleyici; diyet, hijyen, antibiyotik ve biyolojik ajanlar uzun süreli "
        "remisyon sağlar. Cerrahi de gerekebilir."
    ),
    "Hyperhidrosis": (
        "Teşhis: Eli, ayakları veya koltuk altlarını etkileyen aşırı terleme.\n",
        "Bilgi: Sosyal ve mesleki yaşamı etkiler. Antiperspiran kremler, iyontoforez, botulinum "
        "toksini enjeksiyonları veya sinir cerrahisi seçenekleri bulunur."
    ),
    "Milia": (
        "Teşhis: Yüzde 1–2 mm, beyaz veya sarı keratin dolu kistler.\n",
        "Bilgi: Yenidoğanlarda %40’a varan sıklıkla görülür; müdahale gerektirmez. Yetişkinde "
        "travma veya sürfaktan kullanımından sonra ortaya çıkabilir, basit ekstraksiyonla giderilir."
    ),
    "Minocycline Pigmentation": (
        "Teşhis: Minosiklin uzun süreli kullanımında deri, diş ve mukozada mavi-siyah pigment birikimi.\n",
        "Bilgi: İlacın kesilmesi zamanla renk solmasına yol açar; kalıcı skar riski düşüktür. "
        "Alternatif antibiyotik değerlendirilir."
    ),
    "Perioral Dermatitis": (
        "Teşhis: Ağız ve nazolabial bölgede eritemli papül-püstüller, bazen kaşıntılı.\n",
        "Bilgi: Topikal steroidler kötüleştirir. Metronidazol, pimekrolimus veya düşük doz oral "
        "tetrasiklinler etkilidir."
    ),
    "Pomade acne": (
        "Teşhis: Saç ürünlerinin alın bölgesine yayılması sonucu komedon ve papüller.\n",
        "Bilgi: Yağsız, non-komedojenik ürünler tercih edilmeli; bölge temizliği ve topikal "
        "salisilik asit fayda sağlar."
    ),
    "Rosacea": (
        "Teşhis: Yüzde orta hatta eritem, telanjiektazi, papül-püstüller ve bazen göz tutulumu.\n",
        "Bilgi: Alevlendiriciler arasında güneş, sıcak içecek, alkol bulunur. Topikal metronidazol, "
        "azelaik asit ve oral antibiyotiklerle kontrol sağlanır."
    ),
}


@app.route('/predict_acne', methods=['POST'])
def predict_acne():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "no image"}), 400

        file = request.files['image'].read()
        img = Image.open(io.BytesIO(file)).convert('RGB')
        arr = np.array(img.resize((acne_w, acne_h)), dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        acne_interpreter.set_tensor(acne_inp['index'], arr)
        acne_interpreter.invoke()
        preds = acne_interpreter.get_tensor(acne_outp['index'])[0]

        idx   = int(np.argmax(preds))
        label = label_map[idx]
        score = float(preds[idx])
        diag, info = disease_info.get(label, ("", ""))

        return jsonify({
            "label": label,
            "score": score,
            "diagnosis": diag,
            "info": info
        })
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


# --- 3) Üçüncü model (Atopic Dermatitis.tflite) ---
det_interpreter = tf.lite.Interpreter(model_path="Atopic Dermatitis.tflite")
det_interpreter.allocate_tensors()
det_inp, det_out = det_interpreter.get_input_details()[0], det_interpreter.get_output_details()[0]
_, det_H, det_W, _ = det_inp['shape']
det_dtype, (det_scale, det_zp) = det_inp['dtype'], det_inp.get('quantization', (0,0))

label_map_det = {
    0: "Acquired ichthyosis - Adult",
    1: "Atopic Dermatitis Adult Phase",
    2: "Atopic Dermatitis Childhood Phase",
    3: "Atopic Dermatitis Feet",
    4: "Atopic Dermatitis Hyperlinear Creases",
    5: "Ichthyosis vulgaris",
    6: "Keratosis Pilaris",
    7: "Lamellar ichthyosis - Adult",
    8: "Pityriasis Alba"
}

disease_info_det = {
    "Acquired ichthyosis - Adult": (
        "Teşhis: Yetişkinlerde ciltte yaygın pul pul döküntüler ve kalınlaşmış deri görülebilir.",
        "Bilgi: Genetik yatkınlık veya metabolik bozukluklarla ilişkili olabilir; topikal keratolitikler ve yoğun nemlendiriciler uzun süreli kontrol sağlar."
    ),
    "Atopic Dermatitis Adult Phase": (
        "Teşhis: Yetişkinlerde kronik kaşıntılı, inflamatuar eczematoid plaklarla karakterizedir.",
        "Bilgi: Kuru cilt ve alerjenlere aşırı duyarlılık temel rol oynar; cilt bariyerini onaran nemlendiriciler ve düşük-orta potentli topikal kortikosteroidler önerilir."
    ),
    "Atopic Dermatitis Childhood Phase": (
        "Teşhis: Çocukluk döneminde yanak ve eklem kıvrımlarında eritemli, kaşıntılı lezyonlar ile seyreder.",
        "Bilgi: Genetik faktörler ve çevresel tetikleyiciler önemli; nazik cilt bakımı, pimekrolimus ve gerektiğinde sistemik antihistaminikler faydalıdır."
    ),
    "Atopic Dermatitis Feet": (
        "Teşhis: Ayak ve topuk bölgesinde hiperkeratoz, çatlak ve kaşıntılı plaklar görülür.",
        "Bilgi: Nem kaybı ve sürtünme alevlendirir; keratolitik pedler, seramidler içeren kremler ve ayak hijyenine özen tedaviye katkı sağlar."
    ),
    "Atopic Dermatitis Hyperlinear Creases": (
        "Teşhis: Avuç içi ve parmak eklemlerinde derinleşmiş çizgilerle birlikte atopik dermatit tablosu.",
        "Bilgi: Cilt bariyerinin bozulması sonucu oluşur; yoğun nemlendirici uygulamaları ve xerozis kontrolü tedavide anahtar rol oynar."
    ),
    "Ichthyosis vulgaris": (
        "Teşhis: Deride yaygın pul pul dökülme ve derin çatlaklar ile karakterizedir.",
        "Bilgi: Filaggrin gen mutasyonlarına bağlı gelişir; haftalık keratolitik tedavi ve günlük yoğun nemlendirme uzun süreli rahatlama sağlar."
    ),
    "Keratosis Pilaris": (
        "Teşhis: Üst kol ve yanaklarda kıl folikülünde tıkanmaya bağlı küçük, pürüzlü papüller oluşur.",
        "Bilgi: Genellikle ergenlik döneminde belirginleşir; üre veya alfa hidroksi asit içeren losyonlar ve nazik eksfoliasyon etkilidir."
    ),
    "Lamellar ichthyosis - Adult": (
        "Teşhis: Vücutta kalın, lameller pullu plaklar ve yaygın cilt kuruluğu ile seyreder.",
        "Bilgi: Doğuştan gelen genetik bir bozukluktur; oral retinoidler, lipid bazlı nemlendiriciler ve keratolitikler tedavide kullanılır."
    ),
    "Pityriasis Alba": (
        "Teşhis: Yüzde hipopigmente, hafif pul pul döküntüler şeklinde kendini gösterir.",
        "Bilgi: Atopik dermatit alt tipi olabilir; düzenli nemlendirici kullanımı ve düşük potentli topikal steroidlerle semptomlar hafifler."
    ),
}

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / np.sum(e)

@app.route('/predict_atopic', methods=['POST'])
def predict_atopic():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "no image"}), 400

        raw = request.files['image'].read()
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        base = np.array(img.resize((det_W, det_H)), dtype=np.float32) / 255.0

        transforms = [lambda x: x, lambda x: x[:, ::-1, :], lambda x: x[::-1, :, :]]
        preds_sum = np.zeros(len(label_map_det), dtype=np.float32)

        for fn in transforms:
            x_in = fn(base)
            x_in = np.expand_dims(x_in, 0).astype(np.float32)
            if det_dtype == np.uint8 and det_scale > 0:
                x_in = (x_in / det_scale + det_zp).astype(np.uint8)
            det_interpreter.set_tensor(det_inp['index'], x_in)
            det_interpreter.invoke()
            preds_sum += det_interpreter.get_tensor(det_out['index'])[0]

        preds_avg = preds_sum / len(transforms)
        preds = softmax(preds_avg) if not np.allclose(preds_avg.sum(),1,atol=1e-2) else preds_avg

        idx   = int(np.argmax(preds))
        label = label_map_det[idx]
        score = float(preds[idx])
        diag, info = disease_info_det.get(label, ("", ""))

        return jsonify({
            "label": label,
            "score": score,
            "diagnosis": diag,
            "info": info
        })
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


# ------------------------------------------------------------------
# KÖK klasördeki .html, .css, model vs dosyalarını direkt servis eden route
@app.route('/<path:filename>')
def serve_root_files(filename):
    return send_from_directory(BASE_DIR, filename)
# ------------------------------------------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
