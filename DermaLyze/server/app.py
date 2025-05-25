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
        "Teşhis: Deri yağ bezlerinin aşırı sebum üretimi ve gözeneklerin tıkanması sonucu, "
        "Propionibacterium acnes (Cutibacterium acnes) bakterisinin çoğalmasıyla birlikte siyah "
        "nokta (open komedon), beyaz nokta (closed komedon), papül, püstül, nodül ve skarlı lezyonların "
        "karışık formda görüldüğü, inflamatuar ve non-inflamatuar bileşenlerin bir arada bulunduğu kronik "
        "deri hastalığıdır.\n",
        "Bilgi: Ergenlik dönemindeki hormonal değişiklikler başlıca tetikleyicidir; genetik yatkınlık, stres, "
        "beslenme alışkanlıkları ve kozmetik ürün kullanımı da rol oynar. Tedavi; topikal retinoidler, "
        "benzoil peroksit, antibakteriyel losyonlar, sistemik antibiyotikler ve gerektiğinde hormon düzenleyici "
        "ilaç kombinasyonları ile yürütülür."
    ),
    "Acne Infantile": (
        "Teşhis: Yenidoğan ve bebeklik döneminde, maternal hormonların etkisiyle yüz ve gövdede papül-püstüller, "
        "nadir olarak nodül ve kistik lezyonlar formunda ortaya çıkan, genellikle kendiliğinden gerileyen akne türüdür.\n",
        "Bilgi: Çoğunlukla 1–3 aylıkken başlar ve 6 aya kadar geriler. Tedavi gerektirmeyen hafif seyirli bir formdur; "
        "bol temiz hava, nazik cilt bakımı ve sadece gerekli durumlarda dermatolog önerisi ile lokal tedavi uygulanır."
    ),
    "Acne Open Comedo": (
        "Teşhis: Gözenek tıkanıklığındaki oksidasyon sonucu melanin ve lipidlerin koyulaşmasıyla oluşan açık komedonlar "
        "(siyah noktalar) ile karakterizedir; inflamasyon yok denecek kadar azdır.\n",
        "Bilgi: Hafif akne formudur; düzenli eksfoliasyon, salisilik asit içeren tonikler ve nazik temizleyiciler "
        "ile önlenebilir. Derin lezyonlara ilerlemeden erken müdahale önerilir."
    ),
    "Acne Steroid": (
        "Teşhis: Topikal veya sistemik kortikosteroid kullanımına bağlı olarak yüz, gövde ve sırt bölgelerinde "
        "gelişen nodüler ve kistik püstüllerle seyreden, steroide bağlı akneiform dermatit formudur.\n",
        "Bilgi: Steroid dozu azaltılmalı, rejeneratif topikal tedaviler ve alternatif anti-inflamatuar ajanlar "
        "değerlendirilmelidir; kalıcı skarlar için dermatolojik girişimler gerekebilir."
    ),
    "Acne conglobata": (
        "Teşhis: Derin yerleşimli, fistüllü nodüller, abseler ve skar dokusuyla karakterize, en ağır akne formudur.\n",
        "Bilgi: Özellikle genç erişkin erkeklerde görülür; sistemik retinoidler, antibiyotik kombinasyonları ve "
        "biyolojik ajanlar gerekebilir; erken ve agresif tedavi, skar oluşumunu azaltır."
    ),
    "Acne excoriée": (
        "Teşhis: Psikolojik nedenlerle (anksiyete, obsesif davranış) sürekli sıkma ve kaşıma sonucu erozyon, "
        "kızarıklık, skar ve pigment değişiklikleri gelişen, kendini travma ile sürdüren kronik akne formudur.\n",
        "Bilgi: Dermatolog ve psikolog iş birliğiyle hem topikal hem davranışsal terapi yaklaşımları uygulanır."
    ),
    "Acne keloidalis nuchae": (
        "Teşhis: Ense kökündeki kıl foliküllerinin kronik inflamasyonu ve skar dokusu ile keloid benzeri nodüller "
        "oluşmasıdır.\n",
        "Bilgi: Sıkı saç kesimi, travma ve genetik yatkınlık tetikleyebilir; cerrahi eksizyon ve intralezyonal "
        "kortikosteroid enjeksiyonları tedavide etkilidir."
    ),
    "Acne vulgaris": (
        "Teşhis: Papül, püstül, kistik lezyon ve komedonların bir arada bulunduğu, orta şiddetten şiddetliye kadar "
        "farklı şiddet seviyelerinde görülen yaygın akne formudur.\n",
        "Bilgi: Nazik temizleyiciler, topikal retinoidler ve antibiyotikler kombinasyonu ile tedavi edilir; "
        "ağır vakalarda sistemik ajanlar gerekebilir."
    ),
    "Hidradenitis suppurativa": (
        "Teşhis: Aksilla, kasık ve gluteal bölge gibi apokrin bezlerin yoğun olduğu alanlarda kronik, "
        "tekrarlayan ağrılı nodül, apse ve fistüllerle seyreden inflamatuar dermatozdur.\n",
        "Bilgi: Diyet düzenlemesi, hijyen önlemleri, uzun süreli antibiyotikler, biyolojik ajanlar ve gerektiğinde "
        "cerrahi müdahale kullanılır."
    ),
    "Hyperhidrosis": (
        "Teşhis: Ekrin ter bezlerinin aşırı uyarılması sonucu ellerde, ayaklarda veya koltuk altlarında "
        "gözle görülür derecede terleme ataklarının oluştuğu durumdur.\n",
        "Bilgi: Sosyal ve mesleki yaşamı etkiler; antiperspiranlar, iyontoforez, botulinum toksini enjeksiyonları veya "
        "sympathectomy seçenekleri bulunmaktadır."
    ),
    "Milia": (
        "Teşhis: Yüz bölgesinde epidermal keratin tıkaçlarının kistik formda 1–2 mm çapında beyaz ya da sarı "
        "görünümlü kabarcıklar oluşturmasıdır.\n",
        "Bilgi: Yenidoğanlarda fizyolojik, müdahalesiz izlenir; erişkinlerde basit ekstraksiyon veya lazer tedavileri "
        "uygulanabilir."
    ),
    "Minocycline Pigmentation": (
        "Teşhis: Uzun süreli minosiklin kullanımına bağlı olarak deri, diş ve mukozada mavi-siyah renk değişiklikleri "
        "oluşmasıdır.\n",
        "Bilgi: İlacın kesilmesiyle kademeli olarak geriler; kalıcı pigmentasyon nadiren kalıcı skar bırakır."
    ),
    "Perioral Dermatitis": (
        "Teşhis: Ağız çevresi ve nazolabial bölgede eritemli papül-püstüller, bazen hafif kaşıntı ile seyreden "
        "spongioform lezyonlardır.\n",
        "Bilgi: Topikal steroidler durdurulmalı; metronidazol, pimekrolimus veya düşük doz oral tetrasiklinler etkili "
        "olur."
    ),
    "Pomade acne": (
        "Teşhis: Yağlı kozmetik ürünlerin (pomad, pomat) alın ve çene hattına yayılması sonucu komedon ve papül "
        "lezyonlarının gelişmesidir.\n",
        "Bilgi: Non-komedojenik ürünler tercih edilmeli, bölge temizliği ve salisilik asit içeren topikal ajanlar "
        "kullanılmalıdır."
    ),
    "Rosacea": (
        "Teşhis: Yüz merkezi bölgede eritem, teleangiektazi, papül-püstüller ve alevlenme atakları ile birlikte "
        "göz tutulumunun da görülebildiği kronik inflamatuar dermatozdur.\n",
        "Bilgi: Alevlendiriciler arasında güneş, sıcak içecek ve alkol bulunur; topikal metronidazol, azelaik asit "
        "ve oral antibiyotiklerle kontrol sağlanır."
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
        "Teşhis: Yetişkinlerde genellikle hematolojik maligniteler, tiroid bozuklukları veya HIV gibi sistemik hastalıklarla ilişkili, simetrik olarak tüm vücutta ortaya çıkan ince beyaz-gri pul pul döküntüler ve belirgin deri kalınlaşmasıyla karakterizedir. Özellikle ekstremitelerin ekstensor yüzeylerinde yoğunlaşır; kaşıntı ve gerilme hissi eşlik edebilir.\n",
        "Bilgi: Genetik yatkınlık veya metabolik bozukluklarla ilişkili olabilir; topikal keratolitikler ve yoğun nemlendiriciler uzun süreli kontrol sağlar."
    ),
    "Atopic Dermatitis Adult Phase": (
        "Teşhis: Yetişkinlerde kronik, tekrarlayıcı kaşıntılı, inflamatuar eczematoid plaklar şeklinde seyreder. Likenifikasyon (kalınlaşmış deri çizgileri), postinflamatuar hiperpigmentasyon ve fissürler görülebilir; sıklıkla boyun, dirsek içi ve diz arkalarında lokalizedir.\n",
        "Bilgi: Kuru cilt ve alerjenlere aşırı duyarlılık temel rol oynar; cilt bariyerini onaran nemlendiriciler ve düşük-orta potentli topikal kortikosteroidler önerilir."
    ),
    "Atopic Dermatitis Childhood Phase": (
        "Teşhis: Çocukluk döneminde özellikle yanaklar, alın ve eklem kıvrımlarında eritemli, sulu, kaşıntılı lezyonlar şeklinde başlar. Sıklıkla eksüda ve kabuklanma eşlik eder; gece kaşıntısı uykuyu bölebilir.\n",
        "Bilgi: Genetik faktörler ve çevresel tetikleyiciler önemli; nazik cilt bakımı, pimekrolimus ve gerektiğinde sistemik antihistaminikler faydalıdır."
    ),
    "Atopic Dermatitis Feet": (
        "Teşhis: Ayak tabanı ve topuk bölgesinde yoğun hiperkeratoz, derin çatlaklar (fissürler) ve şiddetli kaşıntıyla giden kronik plaklar görülür. Fissürler ağrılı olabilir ve sekundair enfeksiyon riskini artırır.\n",
        "Bilgi: Nem kaybı ve sürtünme alevlendirir; keratolitik pedler, seramidler içeren kremler ve ayak hijyenine özen tedaviye katkı sağlar."
    ),
    "Atopic Dermatitis Hyperlinear Creases": (
        "Teşhis: Avuç içi ve parmak eklemlerinde normalden daha derin ve belirgin çizgiler (hiperlinear kıvrımlar) eşliğinde atopik dermatit bulguları bulunur. Bu çizgiler kuruluk ve çatlak oluşturarak ağrıya yol açabilir.\n",
        "Bilgi: Cilt bariyerinin bozulması sonucu oluşur; yoğun nemlendirici uygulamaları ve xerozis kontrolü tedavide anahtar rol oynar."
    ),
    "Ichthyosis vulgaris": (
        "Teşhis: Deride yaygın, ince pul pul dökülme ve sıkılaşmış, derin çatlaklar ile karakterizedir. Özellikle alt bacaklarda, kolların posterior yüzeyinde ve sırt üstünde belirgindir; palmar hiperlinearizasyon görülebilir.\n",
        "Bilgi: Filaggrin gen mutasyonlarına bağlı gelişir; haftalık keratolitik tedavi ve günlük yoğun nemlendirme uzun süreli rahatlama sağlar."
    ),
    "Keratosis Pilaris": (
        "Teşhis: Üst kol, yanak ve uyluk bölgesinde, folikül girişlerinde çevresel kızarıklıkla birlikte pütürlü, 1–2 mm papüller (tavuk derisi görünümü) şeklinde görülür. Hafif kaşıntı eşlik edebilir.\n",
        "Bilgi: Genellikle ergenlik döneminde belirginleşir; üre veya alfa hidroksi asit içeren losyonlar ve nazik eksfoliasyon etkilidir."
    ),
    "Lamellar ichthyosis - Adult": (
        "Teşhis: Doğuştan itibaren var olan, kalın, yapışkan lameller pullar ve yaygın cilt kuruluğu ile seyreder. Bebeklikte 'collodion baby' tablosu gözlenebilir; pullar koyu kahverengi ve geniş plaklar oluşturur.\n",
        "Bilgi: Doğuştan gelen genetik bir bozukluktur; oral retinoidler, lipid bazlı nemlendiriciler ve keratolitikler tedavide kullanılır."
    ),
    "Pityriasis Alba": (
        "Teşhis: Çocukluk ve adolesan dönemde sıklıkla yüzde, üst kol ve gövde üst kısmında hipopigmente, hafif pul pul döküntüler ve silik sınırlarla karakterizedir. Yaz aylarında daha belirgin hale gelebilir.\n",
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
