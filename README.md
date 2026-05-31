
---

```markdown
# 🤖 Advanced RAG System with Redis Cache & Memory

Bu proje, büyük dil modellerinin (LLM) kurumsal dokümanlar (PDF) ile akıllı ve yüksek performanslı bir şekilde konuşabilmesini sağlayan gelişmiş bir **RAG (Retrieval-Augmented Generation)** uygulamasıdır. 

Projede **Redis** kullanılarak hem konuşma geçmişi (Memory) yönetilmiş hem de mükerrer sorular için LLM maliyetini sıfıra indiren bir önbellekleme (Cache) sistemi kurulmuştur.

## 🚀 Öne Çıkan Özellikler

* **Hibrit Arama (Hybrid Retrieval):** Bilgi getirme doğruluğunu artırmak için kelime tabanlı **BM25** ile vektör tabanlı **FAISS** algoritmaları birlikte çalışır.
* **Akıllı Sorgu Yönetimi:** Kullanıcının girdileri **Question Sanitizer** ve **Query Rewrite** aşamalarından geçerek yapay zekanın en doğru cevabı vermesi sağlanır.
* **Redis Chat Memory:** Oturum bazlı (`SESSION_ID`) konuşma geçmişi tutulur. "Adım neydi?" gibi takip soruları PDF'e gitmeden doğrudan bellekten yanıtlanır.
* **Redis Answer Cache:** Daha önce sorulmuş bir soru tekrar geldiğinde sistem LLM'i çalıştırmaz; cevabı doğrudan Redis cache üzerinden milisaniyeler içinde döner.

---

## 📐 Sistem Mimarisi (Pipeline)

Sistemin çalışma mantığı ve veri akışı şu şekildedir:

Kullanıcı ➔ Question Sanitizer ➔ Memory Router ➔ Redis Chat Memory ➔ Query Rewrite ➔ Hybrid Retrieval (BM25 + FAISS) ➔ Retrieved Docs ➔ Redis Answer Cache ➔ LLM ➔ Answer ➔ Redis'e Yaz

---

## 📁 Proje Yapısı

```text
├── app.py               # Ana uygulama ve pipeline akış kodları
├── memory_manager.py    # Redis üzerinde Chat Memory ve Cache yönetimi
├── docker-compose.yml   # Redis container'ını ayağa kaldıran yapılandırma
├── requirements.txt     # Gerekli Python kütüphaneleri
└── README.md            # Proje dokümantasyonu


## 🛠️ Kurulum ve Çalıştırma

### 1. Redis'i Başlatın

Proje dizininde terminali açıp aşağıdaki komutla Redis container'ını arka planda ayağa kaldırın:

```bash
docker compose up -d



### 2. Bağımlılıkları Yükleyin

Lokal sanal ortamınızı aktif ettikten sonra gerekli paketleri kurun:

pip install -r requirements.txt


### 3. Uygulamayı Çalıştırın

python app.py



## 🔍 Geliştirici Notları (Redis İnceleme)

Redis içerisindeki verileri canlı olarak gözlemlemek için şu komutları kullanabilirsiniz:

# Redis CLI içine girin
docker exec -it redis redis-cli

# Tüm cache ve memory anahtarlarını listeleyin
KEYS *

# Konuşma geçmişini inceleyin
LRANGE chat:session-id 0 -1

```

```

```
