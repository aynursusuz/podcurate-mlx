# Bağımsız ses–metin hizalama

Bu aşama 16 kHz mono, en fazla 30 saniyelik sesi verilen metne hizalar.
Whisper'ın kendi zaman damgalarını kullanmaz. `alignment_status=ok`, geçerli bir
hizalama yolu/zaman dizisi bulunduğunu belirtir; metnin doğru olduğu veya örneğin
eğitim için kabul edilmesi gerektiği anlamına gelmez. `alignment_coverage=1` de
bu doğruluğu kanıtlamaz. Skorlar eşik kalibrasyonu yapılmadan kabul kararı değildir.

| Dil | Akustik model | Çalıştırma | Skor |
|---|---|---|---|
| en, zh, ja | `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` | MLX | `null`: model bu adaptörde CTC/güven skoru sağlamıyor |
| tr | `m3hrdadfi/wav2vec2-large-xlsr-turkish`, FP32 dönüşümü | MLX; CTC trellis NumPy CPU | Blank dahil en iyi CTC yolunun kare başına ortalama log olasılığı |

Qwen revision: `0e1a68e91d815300c7c9754b2a7639378b23db15`.
Türkçe revision: `8699cf317b8f9a834ddf192608324ae5bbd191f0`.
Her iki varsayılan checkpoint Apache-2.0 lisanslıdır.
[Qwen MLX model kartı](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit),
[Türkçe model kartı](https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-turkish).

Qwen'in resmî hizalama modeli 11 dil bildirir; Türkçe bunlardan biri değildir.
Adaptör `tr` girdisini reddeder. Qwen ASR'nin dil listesi farklı bir modelin
özelliğidir. Resmî config'in timestamp sınıf ölçeği 80 ms'dir; bu değer ölçülmüş
sınır hatası değildir. Upstream `fix_timestamp` işlemi sonrasında pozitif süre,
sıralama, ses sınırı ve referans birimlerinin korunması denetlenir.
Upstream düzeltici, en uzun azalmayan alt diziyi kullanır; kısa anomali gruplarını
komşu geçerli zamana taşır, uzun grupları doğrusal ara değerlerle doldurabilir.
Bu işlem ham model çıktısını değiştirir ve eşit başlangıç/bitişler bırakabilir.
Ara değerler gerçek bir sınır ölçümü veya 80 ms sınıflardan daha yüksek doğruluk
kanıtı değildir. Adaptör geçersiz birimleri ayrıca `alignment_invalid_units`
alanında kaydeder; süre uydurarak bunları pozitif hale getirmez.
[Resmî config](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/blob/main/config.json),
[MLX uygulaması](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/qwen3_asr/qwen3_forced_aligner.py).

## Japonca: sinir ağı kullanmayan metin bölümleme

Upstream Qwen processor Japonca için nagisa kullanır. Bu repo, yalnızca ilgili
processor örneğinin `tokenize_japanese` metoduna `MeCabJapaneseUnits` enjekte eder.
Diğer processor işlemleri korunur; site-packages veya global sınıflar değiştirilmez.
`fugashi==1.5.2` ve `unidic-lite==1.0.8` ile MeCab sözlük/maliyet tablosu ve Viterbi
çözümü kullanılır. Nagisa/DyNet varsayılan yolda çağrılmaz. MeCab sinir ağı değildir.
[MeCab'ın algoritma açıklaması](https://taku910.github.io/mecab/),
[Viterbi kaynak kodu](https://github.com/taku910/mecab/blob/master/mecab/src/viterbi.cpp).

Japonca coverage, **MeCab morfolojik birimleri** üzerinden hesaplanır. Bunların
nagisa birimleriyle veya insanın sözcük sınırlarıyla aynı olduğu iddia edilmez.
Sözlük, UniDic 2.1.2'nin değiştirilmiş küçük sürümüdür; yeni adlar ve alan terimleri
için bölümleme hatası oluşabilir. Açık sözlük yolu, paket sürümleri ve kullanılan
sözlük dosyalarının birleşik SHA-256 değeri kimlikte kaydedilir. Güven skoru üretilmez.
fugashi MIT; dağıtılan MeCab ve UniDic sözlükleri BSD lisanslıdır.
[fugashi](https://github.com/polm/fugashi),
[UniDic Lite içeriği ve lisansı](https://github.com/polm/unidic-lite).

## Türkçe dönüşüm ve CTC

MLX encoder/head, MIT lisanslı `mlx-audio` Wav2Vec2/MMS uygulamasından içe aktarılır.
MMS `generate(language=...)` kullanılmaz: bu yol dil parametresini yok sayar.
Türkçe checkpoint'in kendi config'i, sözlüğü ve CTC başlığı yüklenir.
İlk dönüşüm yalnızca `torch.load(weights_only=True, map_location="cpu")` ile ağırlık
okur; üretim çıkarımında PyTorch/ONNX backend'i kullanılmaz. Conv eksenleri ve weight
normalization parametreleri MLX düzenine çevrilir; eğitimde kullanılan SpecAugment
vektörü dışarıda bırakılır. Kalan ağırlıklar `strict=True` yüklenir, dropout kapatılır.
Dönüşüm dosyası kilit altında atomik yazılır; SHA-256 değeri model kimliğine girer.
[İçe aktarılan MLX model kodu](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py).

Ses, kaynak Wav2Vec2 işlemcisiyle aynı tek örnek ortalama/varyans normalizasyonunu
kullanır. CTC emisyonları `[kare, 40]` boyutundadır. Blank ID=0; kare adımı 320 örnek
(20 ms), convolution receptive field 400 örnektir. Zamanlar kare kenarlarından
hesaplanır; bu 20 ms adım insan tarafından ölçülmüş sınır doğruluğu değildir.
Dinamik programlama blank/tekrar geçişlerini içerir ve verilen metnin tamamı için
yol arar. Bir yolun bulunması yanlış referansı otomatik olarak reddetmez.

Türkçe `I/İ` dönüşümü korunur; NFC uygulanır, noktalama kaldırılır. Sayılar veya
sözlük dışı karakterler tahminî okunuşa çevrilmez; `review` döner. Coverage, normalize
referans karakterlerinin hizalanmasıdır. `alignment_score` blank dahil yolun
ortalama log olasılığıdır (≤0); `alignment_token_score` yalnızca nonblank ortalamasıdır.
Blank cezası, metinden çıkarılan sözcüklerin skordan tamamen kaybolmasını engeller.
Bu iki değer de kalibre edilmiş doğruluk olasılığı değildir; dil/model bazında
insan etiketli örneklerle ölçüm gerekir.

## Tekrarlanabilir doğrulama

Normal testler küçük CTC örneklerinin tüm olası yollarını bağımsız tarayarak
trellis sonucuyla karşılaştırır; tekrar eden harf, eksik referans, OOV, zaman
taşması, sıralama ve Japonca bölümleme kontrollerini kapsar. Bunlar model doğruluğu
ölçümü değildir. Gerçek model testleri ayrıca etkinleştirilir:

```sh
PODCURATE_ALIGNMENT_FIXTURES=/path/to/manifest.jsonl \
PODCURATE_ALIGNMENT_CACHE=/path/to/huggingface-cache \
pytest tests/test_alignment_integration.py
```

Manifest satırlarında `audio`, `language`, `reference_text` bulunmalı; ses yolları
manifest dizinine göre çözülür. en/zh/ja/tr örnekleri ve önceden indirilmiş sabit
revision'lar gerekir. Testler kendiliğinden model indirmez. GPU kilidi varsayılan
olarak cache'in üst dizinindeki `gpu.lock`; `PODCURATE_ALIGNMENT_GPU_LOCK` ile
değiştirilebilir. Türkçe test, **aynı gerçek sesler üzerinde özgün PyTorch
checkpoint logits'i ile MLX logits'ini** karşılaştırır. Sayısal dönüşüm toleransları
veri kabul eşikleri değildir.

FLEURS örnekleri kullanıldığında sonuçlar okunmuş konuşma üzerindeki smoke testtir;
ham podcast, konuşmacı çakışması, müzikli kayıt veya bozuk sentetik ses benchmark'ı
değildir. İnsan etiketli sözcük sınırları olmadan sınır doğruluğu raporlanamaz.
[FLEURS veri kaynağı](https://huggingface.co/datasets/google/fleurs).

## 28 Eylül 2026: gerçek Türkçe doğrulaması

M4 Pro, 24 GiB; MLX 0.32.2, mlx-audio 0.5.6, PyTorch 2.14.0,
Transformers 5.17.0. FLEURS test revision'ı
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`; karşılaştırmada aynı FFmpeg ile
çözülmüş float32 örnekler kullanıldı. PyTorch özgün checkpoint'i CPU'da yalnızca
doğrulama amacıyla çalıştırıldı.

| FLEURS kaynak dosyası | Süre (s) | Logits şekli | En büyük mutlak fark | Ortalama mutlak fark | Kare argmax eşleşmesi |
|---|---:|---|---:|---:|---:|
| `10000377651956138413.wav` | 14,16 | 707 × 40 | 0,000179768 | 0,000005309 | %100 |
| `10002518854052673247.wav` | 12,36 | 617 × 40 | 0,000280380 | 0,000005245 | %100 |
| `10067972835475076305.wav` | 11,10 | 554 × 40 | 0,000270367 | 0,000006160 | %100 |
| `10073710776711594036.wav` | 16,62 | 830 × 40 | 0,000204325 | 0,000005375 | %100 |

Dört kayıtta da tam CTC yolu ve sonlu, sıralı sözcük zamanları üretildi. Bunlar
insan etiketli zamanlarla karşılaştırılmadı. Tek çalıştırmadaki hizalama süreleri
sırasıyla 0,808 / 0,183 / 0,164 / 0,250 saniyeydi; ilk yükleme+dönüşüm 4,238 saniye,
MLX'in raporladığı tepe tahsis 2.452.247.624 bayttı. Bu küçük örnek bir kapasite
garantisi veya büyük veri benchmark'ı değildir.

İlk kaydın doğru referansında skor −0,0664; aynı seste üç sözcük çıkarıldığında
−0,5071; üç sözcük tekrarlandığında −0,6282; ilgisiz referansta −1,9304 oldu.
Bu yanlış referanslarda da yol bulunduğu için durum `ok`, coverage 1'di.
Dolayısıyla yalnızca bu alanlarla kalite kabulü yapılamaz. Sayı içeren `Bugün 42
kişi geldi.` referansı açık OOV nedeniyle `review` döndü. Bu deneylerden kabul
eşiği türetilmedi.

İndirilen kaynak ağırlığın SHA-256 değeri:
`f7231dcadd7d585fe2d05a7dc13addf8d6c70d29a6b781f2a4a7c0d42e674225`.
MLX FP32 dönüşümünün SHA-256 değeri:
`85c718fb7aaa048d99c93aed5a78430f5ad33079600a8deb03e73c1b8cd7fe3c`.

## 28 Eylül 2026: Qwen ve Japonca MeCab doğrulaması

Aynı FLEURS revision'ından dil başına dört, toplam 12 gerçek kayıt çalıştırıldı.
Her kayıt zaman damgası üretti; **9 kayıt incelemeye yönlendirildi**. Bu sonuç,
12 kaydın kaliteli hizalandığı veya kabul edildiği anlamına gelmez.

| Dil | Kayıt | `ok` / `review` | Pozitif süreli / toplam birim | Sıfır süreli birim |
|---|---:|---:|---:|---:|
| en | 4 | 0 / 4 | 80 / 88 | 8 |
| zh | 4 | 2 / 2 | 146 / 149 | 3 |
| ja, MeCab birimleri | 4 | 1 / 3 | 115 / 118 | 3 |

`invalid_alignment_timestamp` ve `reference_units_not_fully_aligned` nedenlerinin
her biri 9 kayıtta görüldü; bu iki sayı aynı kayıtları kapsar. Diğer geometri
nedenleri gözlenmedi. Ham timestamp sınıflarında toplam 7 sıfır süreli çift vardı;
upstream `fix_timestamp` sonrasında bu sayı 14 oldu. Dolayısıyla 7 sıfır süre
düzeltici sonrasında oluştu. 80 ms sınıf ölçeği kısa birim sınırlarını güvenilir
biçimde belirlediğini kanıtlamaz; gözlenen tüm sorunları yalnızca bu ölçeğe
atfetmek de doğru değildir. Geçersiz süreler genişletilmedi veya tahmin edilmedi.

Ek kontrolde 12 kaydın her biri için adaptörün pozitif span listesi, doğrudan
aynı MLX modelinin upstream `generate` çıktısındaki pozitif span listesiyle
**birebir eşleşti**. Bu, adaptörün zamanları değiştirmediğini sınar; Qwen portunun
özgün FP32 PyTorch modeline sayısal eşdeğerliği bu çalışmada ölçülmedi. Japonca
MeCab birimleri için nagisa ile eşdeğerlik iddiası da yoktur.

Çalıştırmada nagisa modülü yüklenmedi. MeCab/UniDic sözlük kimliği:
`f829692e3300e8eed8461f237c6557d1c8e298fbce57133f60115b5875414db6`.
Qwen ağırlığının doğrulanan SHA-256 değeri:
`be19ef8ac4326d032e7673342930b14c2df30bd68c1632493b0f563e30829f91`.
İlk model yüklemesi 3,745 saniye; MLX tepe tahsisi 2.449.713.108 bayt.
Kayıt başına ilk çağrı 1,214 saniye, kalan çağrılar 0,101–0,286 saniyeydi.
Bu süreler tek küçük çalıştırmaya aittir; kalite veya kapasite garantisi değildir.
