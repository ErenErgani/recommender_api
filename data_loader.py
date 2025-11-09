import json
import os
import time 
from dotenv import load_dotenv
# --- DOĞRU IMPORT: 'Discover' sınıfı eklendi ---
from tmdbv3api import TMDb, Movie, TV, Discover 
from sentence_transformers import SentenceTransformer
import chromadb
import firebase_admin
from firebase_admin import credentials, firestore

# Posterler için TMDB'nin ana URL'si
TMDB_POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500" 

# --- Yeni mimarimizin adları ---
FIRESTORE_COLLECTION = "content"
CHROMA_COLLECTION = "content_vectors"
BACKUP_FILENAME = "tmdb_content_10k.json"

def main():
    # === 1. ADIM: KURULUM VE ANAHTAR YÜKLEME ===
    print("Dev Veri Yükleyici Script'i başlıyor... .env dosyası yükleniyor.")
    load_dotenv()

    # TMDB API ayarları
    tmdb = TMDb()
    tmdb_api_key = os.getenv('TMDB_API_KEY')
    if not tmdb_api_key:
        print("HATA: TMDB_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin.")
        return
    tmdb.api_key = tmdb_api_key
    tmdb.language = 'en'
    
    movie_api = Movie()
    tv_api = TV()
    # --- DOĞRU İSTEMCİ: Discover API istemcisi oluşturuldu ---
    discover_api = Discover()

    all_content_data = []
    
    # --- BÖLÜM 1: FİLMLERİ ÇEK (5000 Adet) ---
    print("\n--- BÖLÜM 1: FİLMLER ÇEKİLİYOR (5000 Adet) ---")
    for page_num in range(1, 251): 
        print(f"FİLM Sayfası {page_num}/250 çekiliyor...")
        try:
            # --- DOĞRU FONKSİYON: 'discover_api.discover_movies' kullanıldı ---
            discover_params = {
                'page': page_num,
                'sort_by': 'vote_average.desc',
                'vote_count.gte': 500
            }
            discover_movies = discover_api.discover_movies(discover_params)
            
            for m in discover_movies:
                details = movie_api.details(m.id)
                credits = movie_api.credits(m.id)
                
                director = ""
                for crew_member in credits.crew:
                    if crew_member.job == 'Director':
                        director = crew_member.name
                        break
                
                actors_list = [cast.name for i, cast in enumerate(credits.cast) if i < 5]
                poster_url = f"{TMDB_POSTER_BASE_URL}{details.poster_path}" if details.poster_path else ""
                year = details.release_date.split('-')[0] if details.release_date else ""
                
                movie_data = {
                    "id": str(details.id),
                    "type": "movie",
                    "title": details.title,
                    "overview": details.overview,
                    "genres": [g.name for g in details.genres],
                    "director_or_creator": director,
                    "actors": actors_list,
                    "poster_url": poster_url, 
                    "year": year,           
                    "rating": round(details.vote_average, 1) if details.vote_average else 0.0,
                    "runtime": details.runtime if details.runtime else 0
                }
                
                if details.overview:
                    all_content_data.append(movie_data)
                
                time.sleep(0.53) # Hız limiti

        except Exception as e:
            print(f"HATA: Film sayfası {page_num} işlenirken hata: {e}. Bu sayfa atlanıyor.")
            continue 

    print(f"Toplam {len(all_content_data)} adet FİLM bilgisi çekildi.")

    # --- BÖLÜM 2: DİZİLERİ ÇEK (5000 Adet) ---
    print("\n--- BÖLÜM 2: DİZİLER ÇEKİLİYOR (5000 Adet) ---")
    for page_num in range(1, 251): 
        print(f"DİZİ Sayfası {page_num}/250 çekiliyor...")
        try:
            # --- BURASI DA DÜZELTİLDİ: 'discover_api.discover_tv_shows' kullanıldı ---
            discover_params = {
                'page': page_num,
                'sort_by': 'vote_average.desc',
                'vote_count.gte': 500
            }
            discover_tv = discover_api.discover_tv_shows(discover_params) # <-- DOĞRU FONKSİYON
            
            for t in discover_tv:
                details = tv_api.details(t.id)
                credits = tv_api.credits(t.id)
                
                creator = details.created_by[0].name if details.created_by else ""
                actors_list = [cast.name for i, cast in enumerate(credits.cast) if i < 5]
                poster_url = f"{TMDB_POSTER_BASE_URL}{details.poster_path}" if details.poster_path else ""
                year = details.first_air_date.split('-')[0] if details.first_air_date else ""
                runtime = details.episode_run_time[0] if details.episode_run_time else 0

                tv_data = {
                    "id": str(details.id),
                    "type": "tv",
                    "title": details.name,
                    "overview": details.overview,
                    "genres": [g.name for g in details.genres],
                    "director_or_creator": creator,
                    "actors": actors_list,
                    "poster_url": poster_url, 
                    "year": year,           
                    "rating": round(details.vote_average, 1) if details.vote_average else 0.0,
                    "runtime": runtime
                }
                
                if details.overview:
                    all_content_data.append(tv_data)
                
                time.sleep(0.53) # Hız limiti

        except Exception as e:
            print(f"HATA: Dizi sayfası {page_num} işlenirken hata: {e}. Bu sayfa atlanıyor.")
            continue 
            
    print(f"Toplam {len(all_content_data)} adet içerik (film + dizi) bilgisi çekildi.")

    # === 3. ADIM: VERİYİ JSON DOSYASINA YEDEKLEME ===
    print(f"Tüm veriler '{BACKUP_FILENAME}' dosyasına yedekleniyor...")
    with open(BACKUP_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(all_content_data, f, ensure_ascii=False, indent=4)

    # === 4. ADIM: MODELLERİ VE VERİTABANLARINI YÜKLEME ===
    
    print("Firebase'e bağlanılıyor...")
    firebase_key_path = os.getenv('FIREBASE_KEY_PATH')
    if not firebase_key_path:
        print("HATA: FIREBASE_KEY_PATH bulunamadı.")
        return
    
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_key_path)
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        content_collection = db.collection(FIRESTORE_COLLECTION)
    except Exception as e:
        print(f"HATA: Firebase başlatılamadı. Hata: {e}")
        return

    print("ChromaDB başlatılıyor...")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        print(f"Eski ChromaDB koleksiyonu ({CHROMA_COLLECTION}) (varsa) siliniyor...")
        client.delete_collection(name=CHROMA_COLLECTION)
        print("Eski koleksiyon silindi.")
    except Exception:
        print("Eski koleksiyon bulunamadı, bu normal. Devam ediliyor...")
    
    chroma_collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
    
    print("Sentence Transformer modeli yükleniyor... (Model zaten indirildiyse hızlı olacak)")
    model = SentenceTransformer('all-MiniLM-L6-v2') 

    # === 5. ADIM: VERİLERİ VERİTABANLARINA YÜKLEME ===
    print(f"Toplam {len(all_content_data)} içerik veritabanlarına yükleniyor... (Bu işlem ~2+ SAAT sürecek)")
    
    for content in all_content_data:
        content_id_str = content['id']
        
        try:
            # --- FireStore'a Yükleme (Tüm Metin Verileri) ---
            content_collection.document(content_id_str).set(content)

            # --- ChromaDB'ye Yükleme (Vektör Verileri) ---
            vector = model.encode(content['overview']).tolist() 
            
            chroma_collection.add(
                embeddings=[vector],
                documents=[content['overview']],
                metadatas={
                    "title": content['title'], 
                    "type": content['type'],
                    "genres": ", ".join(content['genres'])
                },
                ids=[content_id_str]
            )
            
            print(f"ID: {content_id_str} ({content['title']}) [{content['type']}] işlendi.")
        
        except Exception as e:
            print(f"HATA: İçerik ID {content_id_str} işlenirken hata oluştu. Hata: {e}")
            continue

    print("--- BÜYÜK İŞLEM TAMAMLANDI ---")
    print(f"Tüm {len(all_content_data)} içerik FireStore ve ChromaDB'ye başarıyla yüklendi!")

if __name__ == "__main__":
    main()