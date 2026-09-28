# PodCurate MLX: araştırma ve kanıt kaydı

**Tarama tarihi: 2026-09-28.** Kapsam: İngilizce (`en`), Mandarin/Çince (`zh`), Japonca (`ja`) ve Türkçe (`tr`) gerçek podcast kayıtları ile üretilmiş konuşma seslerinin veri kalitesi. Bu, 24 tekil çalışmadan oluşan odaklı bir taramadır; bütün literatürün incelendiği iddia edilmez. Teknik iddialar birincil makalelere dayanır. “Okuma” alanı kontrol edilen bölümleri belirtir; makalenin tamamının incelendiği anlamına gelmez.

Çalışmaların kendi deney sonuçları ile bu deponun mühendislik tercihleri aşağıda ayrıdır. Başka veri kümelerinde seçilen eşikler, hedef dört dil için doğrulanmış varsayılanlar değildir. Yayınlardaki NVIDIA hızları MLX veya Apple Silicon performans ölçümü olarak kullanılmaz. MLX portunun bulunması da kaynak çalışmadaki sonuçların aynen korunduğunu göstermez.

## Kaynak kanıtları

### R01 — The People’s Speech (2021)

[Makale: A Large-Scale Diverse English Speech Recognition Dataset for Commercial Usage](https://arxiv.org/pdf/2111.09344). **Okuma:** §4.2–4.2.1 ve §5 başlangıcı. Uzun ses ile kusurlu altyazılar, ASR hipotezi ve DSAlign kullanılarak eşlenir; hipotez–transkript CER farkı filtreye girer. İnsan denetimi, filtre sonrasında da segment sınırı ve transkript hataları bulur. İngilizce ASR verisi üzerine sonuçlardır; hizalamanın başarılı olması metnin doğru olduğunu veya Türkçe için aynı eşiğin geçerli olduğunu kanıtlamaz.

### R02 — GigaSpeech (2021)

[Makale: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio](https://arxiv.org/pdf/2106.06909). **Okuma:** §3.3–3.6. Podcast, video ve sesli kitaplarda hizalama/segmentasyondan sonra insertion, deletion ve substitution’a izin veren doğrulama decoder’ı kullanılır. §3.5.3, filler ve disfluency içeren örnekleri atmanın çeşitliliği kısıtladığını belirtir. Çalışma İngilizce ASR corpus’udur; decoder kabul eşiği, her kaydın insan referansına göre gerçek WER garantisi değildir.

### R03 — Emilia (2025; 2024 çalışmasının genişletilmiş sürümü)

[Makale: A Large-Scale, Extensive, Multilingual, and Diverse Dataset for Speech Generation](https://arxiv.org/html/2501.15907v1). **Okuma:** §III ve §IV-A. Ham doğal konuşma için standardizasyon, kaynak ayrıştırma, diarization, VAD, ASR ve kalite filtreleri birleştirilir. Bu, tek konuşmacılı TTS verisi hazırlamaya doğrudan ilgili bir örnektir. En/zh/ja dahil altı dil vardır; tr yoktur. Filtre koşulları değiştirilebilir olarak sunulur. Sekiz RTX 4090 ile ölçülen hız, yerel MLX pipeline’ının hızı değildir.

### R04 — YODAS (ASRU 2023; arXiv 2024)

[Makale: YouTube-Oriented Dataset for Audio and Speech](https://arxiv.org/html/2406.00899v1). **Okuma:** §4.1–4.3. Akustik modelin CTC loss’u ses–altyazı uyumunu filtrelemek için kullanılır. Altyazının konuşma yerine sahne veya müzik açıklaması içerebildiği gösterilir. İngilizce deneyinde daha sıkı eşik küçük performans düşüşü getirir; uzun ve zengin örneklerin elenmesi olası açıklamadır. Türkçe baseline bulunması, Türkçe için evrensel eşik doğrulaması değildir.

### R05 — OWSM v4 (Interspeech 2025)

[Makale: Improving Open Whisper-Style Speech Models via Data Scaling and Cleaning](https://www.isca-archive.org/interspeech_2025/peng25c_interspeech.pdf). **Okuma:** §2.1–2.2, Tables 2–3. YODAS yeniden hizalanır; ses/metin dil tahminlerinin etiketle uyuşması aranır; CTC güveni dil içindeki quantile sıralamasıyla filtrelenir. Yazarlar skorun dile bağlı olduğunu açıkça belirtir. Türkçe değerlendirme vardır; ancak eşik değişiminin etkisi bütün dil ve testlerde aynı değildir. ESPnet/PyTorch sonuçları MLX uygulaması için doğrudan doğrulama değildir.

### R06 — Whisper (2022)

[Makale: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/pdf/2212.04356). **Okuma:** §2.1. Ses–metin dil uyuşması, transkript fuzzy deduplication, değerlendirme verisiyle kopya kontrolü ve yüksek hata/hacimli kaynakların insan denetimi açıklanır. Akustik çeşitlilik ile etiket kalitesi ayrı ele alınır. Bazı konuşmasız örnekler VAD eğitimi için tutulur. Dönemin makine transkripti sezgiselleri, bugünkü bütün sentetik verinin veya sessizliğin reddedilmesi gerektiğini kanıtlamaz.

### R07 — DNSMOS P.835 (2021 preprint; 2022 sürümü)

[Makale: A Non-Intrusive Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors](https://arxiv.org/pdf/2110.01763). **Okuma:** §3–4. Temiz referans gerektirmeden konuşma, arka plan ve genel kaliteyi insan P.835 puanlarından tahmin eder. Eğitim materyali gürültü bastırıcı çıktıları içerir. Ses bozulması için bir ölçümdür; transkript sadakati, konuşmacı kimliği veya Türkçe TTS doğallığını ölçtüğü sonucu çıkarılamaz. Bildirilen korelasyonlar yeni podcast dağılımında doğruluk garantisi değildir.

### R08 — NISQA (Interspeech 2021)

[Makale: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets](https://www.isca-archive.org/interspeech_2021/mittag21_interspeech.pdf). **Okuma:** §2–3. Tek kayıttan genel kalite ve Noisiness, Coloration, Discontinuity, Loudness boyutlarını tahmin eder. Geliştirilen veriler ve testler iletişim/telefon bozulmalarını kapsar; İngilizce kaynaklar ve Almanca konuşma testleri açıklanır. TTS metin sadakati ölçüsü değildir. İncelenen bölümler dört hedef dilde ortak kalibrasyon veya native MLX desteği göstermemektedir.

### R09 — UTMOS (2022)

[Makale: UTokyo-SaruLab System for VoiceMOS Challenge 2022](https://www.isca-archive.org/interspeech_2022/saeki22c_interspeech.pdf). **Okuma:** §2–3 ve ablation tablosu. SSL tabanlı modeller ile daha basit tahmincilerin ensemble’ı sentetik konuşma MOS’unu tahmin eder. Ana benchmark İngilizce, dağılım dışı benchmark Çince ve farklı dinleme testidir; ek insan puanları da toplanır. TTS/voice-conversion algısal değerlendirmesi, podcast transkript doğruluğu testi değildir. Japonca ve Türkçe için ortak kabul eşiği burada doğrulanmaz.

### R10 — UTMOSv2 / The T05 System for the VoiceMOS Challenge 2024 (2024)

[Makale: Transfer Learning from Deep Image Classifier to Naturalness MOS Prediction of High-Quality Synthetic Speech](https://arxiv.org/html/2409.09305v1). **Okuma:** §3.1–3.2. Spectrogram, speech SSL ve veri-domain temsilleri birleştirilerek MOS tahmin edilir. Çalışma, görülmemiş dinleme veri setleri için domain encoding’in düzgün çalışmayabileceğini açıkça belirtir ve puan aralığı yanlılığını inceler. Dolayısıyla skor, her dil/üretici için mutlak doğal konuşma standardı veya doğrudan kabul eşiği değildir.

### R11 — CTC-Segmentation (2020)

[Makale: CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition](https://arxiv.org/pdf/2007.09127). **Okuma:** §3.1–3.2 ve §4.1. CTC modelinden frame posterior’larıyla dinamik programlama üzerinden ses–metin hizası çıkarılır. Alt pencere ortalamalarının minimumundan türetilen segment skoru, uzun örnekteki yerel uyumsuzluğu da cezalandırır. Dil sözlüğü ve normalization önemlidir. Almanca corpus ve TEDlium değerlendirmesi, Türkçe başarısı veya hizalanan metnin bağımsız doğruluk garantisi değildir.

### R12 — WhisperX (Interspeech 2023)

[Makale: Time-Accurate Speech Transcription of Long-Form Audio](https://www.isca-archive.org/interspeech_2023/bain23_interspeech.pdf). **Okuma:** §2.4–2.7 ve §3.1–3.2. VAD Cut & Merge, batch ASR ve harici alignment modeliyle kelime zamanları üretilir; WER yanında insertion, tekrar ve zamanlı kelime doğruluğu değerlendirilir. Hedef dile uygun alignment modeli gerekir. Yazarlar çok dilli kelime segmentasyonu için nicel non-English değerlendirme yapamadıklarını belirtir. MLX Whisper, tek başına bütün WhisperX işlevlerini sağlamaz.

### R13 — Careless Whisper (2024)

[Makale: Speech-to-Text Hallucination Harms](https://arxiv.org/html/2402.08021v1). **Okuma:** §2.3–2.4, §3.1 ve ilgili sonuçlar. Aynı sesin API çıktıları arasındaki farklarla aday örnekler seçilir, insan incelemesiyle sesin içermediği ifadeler saptanır. Uzun konuşmasız sürelerle hallucination ilişkisi raporlanır. Deneyler belirli AphasiaBank görüşmeleri ve 2023 API sürümlerine aittir; güncel yerel model veya Türkçe hata oranını vermez. Doğal duraklama tek başına kusur olarak etiketlenemez.

### R14 — Optimized Synthetic Data Generation (2025)

[Makale: Towards Improved Speech Recognition through Optimized Synthetic Data Generation](https://arxiv.org/html/2508.21631v1). **Okuma:** §2.3, §3.1–3.4 ve §4 başlangıcı. TTS üretim metni, üretilen sesin ASR transkriptiyle WER üzerinden karşılaştırılır; sorunlu örnekler sınırlı sayıda yeniden üretilir. Québec Fransızcasında süre kontrolünden daha etkili sonuçlar raporlanır. Verifier hatası her zaman generator hatası değildir; daha iyi intrinsik kalite skorları da her durumda daha iyi downstream ASR anlamına gelmez.

### R15 — Seed-TTS (2024)

[Makale: A Family of High-Quality Versatile Speech Generation Models](https://arxiv.org/html/2406.02430). **Okuma:** §3.1–3.2 ve değerlendirme tabloları. İçerik için ASR-WER, konuşmacı için referans sesle embedding benzerliği ve insan değerlendirmesi ayrı kullanılır. En/zh için farklı ASR modelleri seçilir. Daha düşük WER’in daha iyi öznel konuşmacı benzerliği anlamına gelmediği; aksan/ifadenin düzleşebildiği belirtilir. WER=0 koşulu veya ja/tr’ye aynı eşik aktarımı desteklenmez.

### R16 — F5-TTS (2024 preprint; ACL 2025)

[Makale: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://arxiv.org/html/2410.06885), [yayın kaydı](https://aclanthology.org/2025.acl-long.313/). **Okuma:** değerlendirme düzeni, Ek B.5–B.6 ve C’nin ilgili bölümleri. Yanlış telaffuz, kelime atlama, tekrar, konuşmacı benzerliği ve UTMOS ayrı değerlendirilir. ASR modeli veya konuşmacı prompt’u değişince ölçülen WER değişebilir. Bu, WER farkının bütünüyle TTS hatası olmadığını gösterir; genel bir filtre tarifi veya dört dil için kalibrasyon değildir.

### R17 — CosyVoice 3 (2025)

[Makale: Towards In-the-wild Speech Generation via Scaling-up and Post-training](https://arxiv.org/html/2505.17589). **Okuma:** §2.3–2.4, §3 ve §4.3–4.4. Diarization, VAD, ses olayları, kesilmiş kelime uçları, çoklu ASR arasındaki pairwise WER, forced alignment ve ses/metin uzunluk oranları veri hazırlamada kullanılır. En/zh/ja kapsamı vardır, tr yoktur. ASR mutabakatı gerçek doğruluk garantisi değildir; çalışma eşikleri doğrudan aktarılamaz. Gürültü azaltma, kaliteyi ölçmekten ayrı bir ses dönüşümüdür.

### R18 — Qwen3-TTS Technical Report (2026)

[Rapor](https://arxiv.org/html/2601.15621). **Okuma:** §3.2 ve §4.2.1–4.2.6. İçerik tutarlılığı WER/CER, konuşmacı benzerliği ayrı skorlarla değerlendirilir; uzun üretimde tekrar, atlama ve prozodi süreksizliği incelenir. Yüksek kaliteli veri için özel pipeline belirtilir ancak §3.2 tam algoritma ve eşikleri açıklamaz. Bu nedenle filtreyi aynen yeniden üretme iddiası kurulamaz. En/zh/ja vardır, tr yoktur; CUDA/vLLM gecikmeleri MLX ölçümü değildir.

### R19 — Towards Selection of Text-to-speech Data to Augment ASR Training (2023)

[Makale](https://arxiv.org/html/2306.00998). **Okuma:** §1–3. Gerçek ve sentetik sesi ayırmayı öğrenen GRU temsilleriyle sentetik alt küme seçilir; amaç gerçek ses üzerindeki downstream ASR başarısıdır. Gerçek veriye en çok benzeyen örneklerin her zaman en yararlı eğitim verisi olması desteklenmez; dilsel çeşitlilik ile akustik bozulma ayrılmalıdır. LibriSpeech/İngilizce sonuçları TTS eğitimi, podcast dinlenebilirliği veya ja/zh/tr başarısına doğrudan genellenemez.

### R20 — TTSDS2 (2025 preprint; ICLR 2026)

[Makale: Resources and Benchmark for Evaluating Human-Quality Text to Speech Systems](https://arxiv.org/html/2506.19441). **Okuma:** §2, faktör/dağılım tanımları ve abstract. Sentetik ve gerçek ses kümelerinin temsilleri; genel kalite, konuşmacı, prozodi ve anlaşılabilirlik faktörlerinde karşılaştırılır. Yöntem bir dağılım metriğidir: tek kayda iyi/kötü eşiği atayan filtre değildir. Filtre öncesi/sonrası çeşitlilik değerlendirmesine adaydır. Dört hedef dilin tamamında kalibrasyon ve yerel MLX maliyeti bu taramada doğrulanmadı.

### R21 — Vox-Evaluator (2025 preprint; AAAI 2026)

[Makale: Enhancing Stability and Fidelity for Zero-shot TTS with a Multi-Level Evaluator](https://arxiv.org/html/2510.20210), [yayın kaydı](https://ojs.aaai.org/index.php/AAAI/article/view/40636). **Okuma:** model görevi, “Dataset for Vox-Evaluator” ve deneyler. Yanlış telaffuz, atlama, tekrar, anormal duraklama ve gürültünün zaman aralıkları ile bütün kayıt kalitesi ayrı çıktılar olarak öğrenilir. İncelenen deneyler İngilizce veri üzerindedir. Dört dil için hazır doğrulama, erişilebilir ağırlıkların kapsamı ve native MLX portu bu okumayla kurulmaz.

### R22 — Raon-OpenTTS (2026)

[Makale: Open Models and Data for Robust Text-to-Speech](https://arxiv.org/html/2605.20830v1). **Okuma:** §3.1–3.2 ve Table 3 ablation. DNSMOS, Whisper-small WER ve Silero speech ratio sıralamaları birleştirilir. Sabit update sayılı İngilizce TTS deneyinde %15 örnek eleme en iyi ortalama rank’ı verir; %50 eleme genellikle daha zayıftır. Bütün metriklerde üstünlük yoktur. Bu, kontrollü filtre ablation’ı için doğrudan kanıttır; aynı yüzdeleri dört dile veya bu depoya varsayılan yapmak için kanıt değildir.

### R23 — WenetSpeech4TTS (2024)

[Makale: A 12,800-hour Mandarin TTS Corpus for Large Speech Generation Model Benchmark](https://arxiv.org/html/2406.05763). **Okuma:** §2.1–2.6, §3 ve §4.2–4.4. Mandarin podcast/video verisinde segment birleştirme, sınır genişletme, enhancement, çoklu konuşmacı tespiti ve yeniden transkripsiyon uygulanır. Kalite alt kümeleri **DNSMOS P.808** ile oluşturulur; bu skor R07’deki **P.835** ile aynı metrik kabul edilmez. TTS eğitimi ve devam eğitimiyle doğrulanan sonuçlar, dört dile taşınabilir eşikler veya bu deponun aynı pipeline’ı uyguladığı anlamına gelmez.

### R24 — MLS (2020)

[Makale: A Large-Scale Multilingual Dataset for Speech Research](https://arxiv.org/pdf/2012.03411). **Okuma:** §3.2, §3.5, §4.1 ve §4.3. Sesli kitaplarda ASR pseudo-label’ları kitap metnine TF-IDF ve Smith–Waterman ile eşlenir; aday transkript–hipotez WER’iyle filtrelenir. Eğitim/geliştirme/test ayrımında konuşmacı çakışması engellenir; geliştirme/test metinleri insan tarafından düzeltilir. Pseudo-label’ın yanlış olabileceği açıkça belirtilir. Corpus sekiz Avrupa dili içerir; zh/ja/tr doğrulaması veya doğal podcast dağılımına doğrudan aktarım göstermez.

## Bu deponun tasarım tercihleri ve mevcut sınırı

Aşağıdaki tercihler yayınların birebir replikasyonu değildir. Mevcut çekirdek, ses–metin çiftlerinin yerel puanlanmasını ve sonuçların denetlenebilir olmasını hedefler; ham podcast’ten doğrulanmış TTS/ASR corpus’una uzanan bütün aşamaları tamamlamaz.

| Mevcut kapsam | Ölçtüğü veya yaptığı | Sınırı |
|---|---|---|
| MLX Whisper ve verilen referans metinle CER | Dört hedef dilde ASR çıktısı–metin farkı | Referans metin ve ASR de hatalı olabilir; gerçek hata oranının bağımsız ölçümü değildir |
| `en` / `tr` için WER | Kelime düzeyinde metin farkı | `zh` / `ja` için aynı kelime ayırma varsayımı uygulanmaz; CER de dört dil arasında doğrudan karşılaştırılabilir kalite standardı değildir |
| İsteğe bağlı DNSMOS, CPU backend | Akustik kaliteye ilişkin ek skor | Native MLX değildir; içerik sadakati ve konuşmacı benzerliği değildir |
| Birebir PCM kopya denetimi | Aynı PCM içerikli kayıtları belirleme | Yakın kopya, yeniden kodlanmış ses, tekrar yayımlanmış bölüm veya aynı metnin farklı seslendirmelerini bütünüyle çözmez |
| İsteğe bağlı MLX Silero hazırlama | Konuşma bölgelerinden aday segment hazırlama | Diarization, overlap ayrıştırma veya kelime hizalama değildir |
| Varsayılan olarak yalnız skor; ayrı kullanıcı eşikleri | Ölçümü kabul/red politikasından ayırma | Veriyle kalibre edilmedikçe eşiklerin kalite faydası doğrulanmış değildir |

### Güçlü bir pipeline için henüz karşılanmayan gereksinimler

Bu gereksinimler hedef kullanım ve kaynak kapsamına göre doğrulanmalıdır; hepsi her ASR/TTS hedefinde aynı ret politikasını gerektirmez.

- **Podcast konuşmacı ve overlap denetimi:** diarization uygulanmış değildir. Tek konuşmacılı TTS çıktısı için speaker/overlap bilgisi gerekir; VAD bunu sağlamaz. Dayanak: R03, R17.
- **Metin–ses hizası ve kesilmiş uçlar:** forced alignment uygulanmış değildir. Yanlış zaman sınırları, kısmi sözcükler ve yerel atlamalar yalnız kayıt geneli CER/WER ile bütünüyle tanımlanamaz. Dayanak: R01, R02, R11, R12, R17.
- **Konuşmacı sadakati:** güvenilir referans sesle speaker similarity uygulanmış değildir. Sentetik sesin hedef konuşmacıyı koruması içerik doğruluğundan ayrı değerlendirme gerektirir. Dayanak: R15, R16, R18.
- **Müzik/gürültü ve dönüşüm kaydı:** kaynak ayrıştırma uygulanmış değildir. R03/R17’deki ayrıştırma veya gürültü azaltma, orijinal ses üzerinde değişikliktir; çıktı bozulmaları ayrıca değerlendirilmelidir. Bu işlem bir kalite skoru yerine geçmez.
- **Dil ve bağımsız doğrulayıcı:** ses/metin LID uyuşması ve çoklu ASR kıyası mevcut çekirdeğin parçası değildir. Özellikle hatalı altyazı ve code-switching için ayrı değerlendirme gerekir. Dayanak: R05, R06, R17.
- **Kalibrasyon ve downstream kanıt:** dil, gerçek/sentetik kaynak, üretici, konuşmacı ve ifade tarzına göre insan denetimli örnekler; saklanan/eliminasyon adayı veride hata ve çeşitlilik incelemesi; gerçek held-out veri üzerinde filtre ablation’ı henüz gerekli çalışmalardır. R04, R14, R19, R20, R22 daha sıkı filtrenin otomatik olarak daha iyi olmadığını veya kalitenin birden çok eksende ölçülmesini destekler.

Bu açık işler nedeniyle mevcut çekirdek “tamamlanmış kaliteli veri pipeline’ı” veya doğrulanmış dört dilli filtre olarak sunulmaz. Kaynak sonuçları, bu deponun filtre sonrasında eğitim kalitesini artırdığının ölçümü değildir.

## Kapsam dışı bırakılanlar

- **WAVe:** [yayınevi kaydı](https://www.sciencedirect.com/science/article/pii/S0020025526005220) ve yazar README’si bulundu; tam metin erişimi başarısız/403 oldu. Sayı tarihi 5 Ekim 2026, tarama tarihinden sonradır; online-first tarihi doğrulanmadı. Ana 24 kaynak arasında değildir ve uygulama kararı için dayanak alınmadı.
- **Metin/SFT filtreleme çalışmaları:** DCLM, FineWeb, DEITA ve AlpaGasus, sentetik konuşma sesi filtresinin doğrudan kanıtı olarak kullanılmadı.
- **Yalnız abstract düzeyindeki ek adaylar:** kaynak sayısını artırmak için ana matrise eklenmedi. Sosyal medya, üçüncü taraf özetleri ve kopya model repoları teknik yöntem kanıtı sayılmadı.

## Kısa arama günlüğü

Arama sonuçlarından makalelerin yöntem, değerlendirme ve sınırlılık bölümlerine geçildi; ilgili kaynakların atıfları takip edildi. Temel sorgu aileleri:

1. `Emilia GigaSpeech People's Speech YODAS speech data filtering alignment`
2. `DNSMOS P.835 NISQA UTMOS UTMOSv2 speech quality prediction`
3. `CTC segmentation WhisperX forced alignment hallucination Careless Whisper`
4. `synthetic speech data filtering ASR consistency WER CER TTS quality`
5. `Seed TTS F5 TTS CosyVoice 3 Qwen3 TTS data filtering speaker similarity`
6. `TTSDS2 Vox-Evaluator synthetic speech evaluation`
7. `WAVe word-aligned verification synthetic speech` — kapsam dışı erişim/tarih kaydı.
8. `Raon-OpenTTS 2605.20830` — §3.2 ve filtre ablation tablosu ayrıca doğrulandı.
9. Verilen birincil kaynaklara doğrudan erişim: WenetSpeech4TTS `2406.05763` §2–4 ve MLS `2012.03411` §3.2, §3.5, §4.1, §4.3. P.808/P.835 ayrımı, transcript retrieval ve insan doğrulaması kontrol edildi.

Tamamlanmamış alanlar: hedef dillere özgü bağımsız kalibrasyon literatürü, lehçe/konuşma bozukluğu alt gruplarında filtre yanlılığı, yakın ses kopyalarının temizlenmesi ve bütün adayların MLX port eşdeğerliği. Bu kayıt bu alanların çözüldüğü anlamına gelmez.
