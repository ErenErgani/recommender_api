# -*- coding: utf-8 -*-
import os
import json 
from flask import Flask, jsonify, request, Response 
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np 
import traceback # Hata ayıklama için

# --- Veritabanı Adları ---
FIRESTORE_COLLECTION = "content"
CHROMA_COLLECTION = "content_vectors"

# --- Ayar (Tuning) Parametreleri ---
CANDIDATE_POOL_SIZE = 1500  # Aday Havuzu
MIN_SCORE_THRESHOLD = 70.0 # Ana Sayfa Kalite Eşiği
CHATBOT_DISCOVERY_THRESHOLD = 50.0 # Chatbot "Keşif" Eşiği (AI+Virality)

# --- Chatbot Tür Eşleştirme Sözlüğü (Daha Akıllı Sıralama) ---
GENRE_MAP = {
    # Önce uzun (spesifik) olanlar
    "bilim kurgu": "Science Fiction",
    "tv filmi": "TV Movie",
    # Sonra kısa (genel) olanlar
    "aksiyon": "Action", "macera": "Adventure", "animasyon": "Animation",
    "komedi": "Comedy", "suç": "Crime", "belgesel": "Documentary",
    "dram": "Drama", "aile": "Family", "fantastik": "Fantasy",
    "tarih": "History", "korku": "Horror", "müzik": "Music",
    "gizem": "Mystery", "romantik": "Romance", 
    "gerilim": "Thriller", "savaş": "War",
    "western": "Western"
}

# --- 1. KURULUM VE BAĞLANTI ---
print("Sunucu başlatılıyor... .env dosyası yükleniyor.")
load_dotenv()
app = Flask(__name__)

# (Firebase, ChromaDB, Model Yükleme kodları... Değişiklik yok)
print("Firebase'e bağlanılıyor...")
firebase_key_path = os.getenv('FIREBASE_KEY_PATH')
if not firebase_key_path:
    print("HATA: FIREBASE_KEY_PATH bulunamadı.")
else:
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_key_path)
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        content_collection = db.collection(FIRESTORE_COLLECTION)
        users_collection = db.collection('users') 
        print(f"Firebase bağlantısı başarılı. ({FIRESTORE_COLLECTION} ve users bağlı)")
    except Exception as e:
        print(f"HATA: Firebase başlatılamadı. Hata: {e}")

print("ChromaDB'ye bağlanılıyor...")
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = client.get_collection(name=CHROMA_COLLECTION)
    print(f"ChromaDB bağlantısı başarılı. Koleksiyonda {chroma_collection.count()} adet içerik vektörü bulundu.")
except Exception as e:
    print(f"HATA: ChromaDB koleksiyonu ('{CHROMA_COLLECTION}') bulunamadı. Hata: {e}")

print("Sentence Transformer modeli yükleniyor...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence Transformer modeli yüklendi.")
except Exception as e:
    print(f"HATA: Sentence Transformer modeli yüklenemedi. Hata: {e}")


# --- 2. YARDIMCI FONKSİYONLAR ---
# (get_content_from_firestore, extract_ids_from_entries... Değişiklik yok)
def get_content_from_firestore(ids_list):
    if not ids_list:
        return {}
    content_data = {}
    unique_ids = list(set(ids_list))
    for i in range(0, len(unique_ids), 30):
        chunk_ids = unique_ids[i:i+30]
        try:
            docs = content_collection.where(u"__name__", 'in', chunk_ids).stream()
            for doc in docs:
                content_data[doc.id] = doc.to_dict()
        except Exception as e:
            print(f"FireStore'dan '{chunk_ids}' çekilirken hata: {e}") 
            continue
    return content_data

# 'normalize_score' fonksiyonu artık iki modlu
def normalize_content_score(value, max_points=30, min_val=0, max_val=2.0):
    """ ChromaDB mesafesini (0-2) 0-max_points arası puana çevirir """
    if (max_val - min_val) == 0: return 0.0 
    normalized = (value - min_val) / (max_val - min_val)
    score = (1 - normalized) * max_points
    return max(0, min(score, max_points))

def extract_ids_from_entries(entries, entry_type):
    ids = []
    if not entries:
        return ids
    for entry in entries:
        if entry.get('id') and entry.get('type'): 
            ids.append(str(entry.get('id')))
    return ids

# --- YENİ: Chatbot için "Kalite/Virality" Puanı ---
def get_virality_score(rating, max_points=50):
    """ 
    İçeriğin genel puanına (rating) göre 0-50 arası kademeli 'Virality Puanı' verir.
    """
    if rating >= 9.0:
        return max_points # 50 Puan
    elif rating >= 8.5:
        return max_points * 0.8 # 40 Puan
    elif rating >= 8.0:
        return max_points * 0.6 # 30 Puan
    elif rating >= 7.5:
        return max_points * 0.4 # 20 Puan
    elif rating >= 7.0:
        return max_points * 0.2 # 10 Puan
    else:
        return 0 # 0 Puan


# --- 3. TEST UÇ NOKTASI (ENDPOINT) ---
@app.route('/')
def index():
    data = {"message": f"API Sunucusu çalışıyor! (Tuning 8.0 - Kademeli/Keşif Motorlu)"}
    json_response = json.dumps(data, ensure_ascii=False, indent=4)
    return Response(json_response,
                    content_type="application/json; charset=utf-8")

# --- 4. ANA ÖNERİ UÇ NOKTASI (BUNA DOKUNULMADI, GÜVENDE) ---
@app.route('/api/v1/recommendations', methods=['GET'])
def get_recommendations():
    
    user_id = request.args.get('userId')
    content_type_filter = request.args.get('type')
    
    if not user_id:
        return jsonify({"error": "Kullanıcı ID'si (userId) gerekli."}), 400
    
    print(f"\n--- Yeni Öneri İsteği (ANA MOTOR): {user_id} | Tip Filtresi: {content_type_filter} ---")

    try:
        # (Bu fonksiyonun tüm içeriği BİR ÖNCEKİ KODLA AYNIDIR, GÜVENDE)
        # 1. KULLANICI LİSTELERİNİ ÇEK
        user_doc_ref = users_collection.document(user_id)
        user_doc = user_doc_ref.get()
        if not user_doc.exists:
            return jsonify({"error": f"Kullanıcı ({user_id}) bulunamadı."}), 404
        user_data = user_doc.to_dict()
        fav_ids = extract_ids_from_entries(user_data.get('favoritesEntries', []), "favorite")
        watched_ids = extract_ids_from_entries(user_data.get('watchedEntries', []), "watched")
        watchlist_ids = extract_ids_from_entries(user_data.get('watchlistEntries', []), "watchlist")
        all_user_ids = set(fav_ids + watched_ids + watchlist_ids)
        if not all_user_ids:
            return jsonify({"message": "Öneri için profilinizde yeterli veri (favori, izlenen vb.) bulunamadı."}), 200
        print(f"Tüm Favori ID'leri (Ağırlıklandırma için): {list(all_user_ids)}")

        # 2. TÜM LİSTELERİN VEKTÖRLERİNİ VE AĞIRLIKLARINI HAZIRLA
        all_ids_to_fetch = list(all_user_ids)
        content_meta_data = get_content_from_firestore(all_ids_to_fetch)
        valid_ids = [id for id in all_ids_to_fetch if id in content_meta_data]
        if not valid_ids:
            print(f"Uyarı: Kullanıcının listelerindeki ID'ler ({all_ids_to_fetch}) bizim veritabanımızda bulunamadı.")
            return jsonify({"message": "Listenizdeki içerikler, öneri veritabanımızdaki içeriklerle eşleşmedi."}), 200
        print(f"Geçerli favori ID'ler (veritabanında bulunan): {valid_ids}")
        vector_data = chroma_collection.get(ids=valid_ids, include=['embeddings'])
        id_to_vector_map = {id: np.array(emb) for id, emb in zip(vector_data['ids'], vector_data.get('embeddings', []))}

        # 3. AĞIRLIKLI "ZEVK PROFİLİ VEKTÖRÜ" HESAPLA
        all_vectors = []
        weights = []
        for id_str in valid_ids:
            if id_str in id_to_vector_map:
                all_vectors.append(id_to_vector_map[id_str])
                if id_str in fav_ids: weights.append(1.0)
                elif id_str in watched_ids: weights.append(0.75)
                elif id_str in watchlist_ids: weights.append(0.25)
        if not all_vectors: return jsonify({"error": "Geçerli içerikler için vektör bulunamadı."}), 500
        taste_vector = np.average(all_vectors, axis=0, weights=weights)
        print(f"Ağırlıklı zevk profili {len(all_vectors)} vektör (Ağırlıklar: {weights.count(1.0)}F, {weights.count(0.75)}Wd, {weights.count(0.25)}Wl) üzerinden hesaplandı.")

        # 4. İÇERİK TABANLI (ADAY) İÇERİKLERİ ÇEK
        chroma_filter = None
        if content_type_filter in ['movie', 'tv']:
            chroma_filter = {"type": content_type_filter}
        
        print(f"ChromaDB'den {CANDIDATE_POOL_SIZE} adet aday çekiliyor...")
        query_results = chroma_collection.query(
            query_embeddings=[taste_vector.tolist()],
            n_results=CANDIDATE_POOL_SIZE, 
            where=chroma_filter, 
            include=['distances'] 
        )
        candidate_ids = query_results['ids'][0]
        distances = query_results['distances'][0]
        cand_content_data = get_content_from_firestore(candidate_ids)

        # 5. KURAL TABANLI (%70) PUANLAMA
        print(f"Kural tabanlı puanlama {len(candidate_ids)} aday için başlıyor...")
        
        strong_signal_ids = set(fav_ids + watched_ids)
        strong_signal_data = {id: data for id, data in content_meta_data.items() if id in strong_signal_ids}
        fav_creators = set(data.get('director_or_creator', '') for data in strong_signal_data.values() if data.get('director_or_creator'))
        fav_genres = set(genre for data in strong_signal_data.values() for genre in data.get('genres', []))
        fav_actors = set(actor for data in strong_signal_data.values() for actor in data.get('actors', [])[:3])
        final_scored_recommendations = []
        
        for i, cand_id in enumerate(candidate_ids):
            if cand_id in all_user_ids: continue 
            if cand_id not in cand_content_data: continue
            cand_content = cand_content_data[cand_id]
            
            # Ana Motor: 30/70 Puanlama
            content_score = normalize_content_score(distances[i], max_points=30) # Max 30
            rule_score = 0
            
            # Kademeli Puanlama (Max 70)
            if cand_content.get('director_or_creator') and cand_content.get('director_or_creator') in fav_creators:
                rule_score += 30
            cand_actors = set(cand_content.get('actors', [])[:3])
            actor_matches = len(cand_actors.intersection(fav_actors))
            if actor_matches == 1:
                rule_score += 15
            elif actor_matches >= 2:
                rule_score += 20
            cand_genres = set(cand_content.get('genres', []))
            genre_matches = len(cand_genres.intersection(fav_genres))
            if genre_matches == 1:
                rule_score += 5
            elif genre_matches == 2:
                rule_score += 10
            elif genre_matches >= 3:
                rule_score += 15
            if cand_content.get('rating', 0) >= 8.0:
                rule_score += 5
            
            final_score = content_score + rule_score
            
            final_scored_recommendations.append({
                "content_id": cand_id,
                "type": cand_content.get('type'),
                "title": cand_content.get('title'),
                "poster_url": cand_content.get('poster_url'),
                "year": cand_content.get('year'),
                "final_score": round(final_score, 2),
                "debug_details": {
                    "content_score (max 30)": round(content_score, 2),
                    "rule_score (max 70)": rule_score
                }
            })

        # 6. SONUÇLARI SIRALA VE FİLTRELE
        sorted_recommendations = sorted(
            final_scored_recommendations, 
            key=lambda x: x['final_score'], 
            reverse=True
        )
        high_quality_recommendations = [
            rec for rec in sorted_recommendations 
            if rec['final_score'] >= MIN_SCORE_THRESHOLD
        ]
        top_recommendations = high_quality_recommendations[:10]
        print(f"Toplam {len(sorted_recommendations)} adaydan, {len(high_quality_recommendations)} tanesi {MIN_SCORE_THRESHOLD} puan eşiğini geçti. İlk {len(top_recommendations)} tanesi döndürülüyor.")
        
        json_response = json.dumps(top_recommendations, ensure_ascii=False, indent=4)
        return Response(json_response,
                        content_type="application/json; charset=utf-8")

    except Exception as e:
        print(f"HATA: Öneri hesaplanırken bir sorun oluştu: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Sunucu hatası: Öneri hesaplanamadı."}), 500

# --- 5. CHATBOT UÇ NOKTASI (BURASI TAMAMEN DEĞİŞTİ) ---
@app.route('/api/v1/chatbot', methods=['GET'])
def get_chatbot_recommendations(): 
    
    user_id = request.args.get('userId')
    content_type_filter = request.args.get('type')
    query = request.args.get('query')
    
    if not user_id:
        return jsonify({"error": "Kullanıcı ID'si (userId) gerekli."}), 400
    
    # --- SORGUDAN TÜR FİLTRESİ OLUŞTUR ---
    genre_filters = []
    if query:
        print(f"Chatbot Sorgusu Alındı: '{query}'")
        lower_query = query.lower()
        sorted_genre_keys = sorted(GENRE_MAP.keys(), key=len, reverse=True)
        for key in sorted_genre_keys:
            if key in lower_query:
                genre_filters.append(GENRE_MAP[key])
                lower_query = lower_query.replace(key, "") 
        if genre_filters:
            print(f"Sorgudan bulunan Tür Filtreleri: {genre_filters}")
            
    print(f"\n--- Yeni Öneri İsteği (CHATBOT): {user_id} | Tip Filtresi: {content_type_filter} | Tür Filtreleri: {genre_filters} ---")

    try:
        # 1. KULLANICI LİSTELERİNİ ÇEK (Sadece 'görmezden gelmek' için)
        user_doc_ref = users_collection.document(user_id)
        user_doc = user_doc_ref.get()
        if not user_doc.exists:
            return jsonify({"error": f"Kullanıcı ({user_id}) bulunamadı."}), 404
        user_data = user_doc.to_dict()
        fav_ids = extract_ids_from_entries(user_data.get('favoritesEntries', []), "favorite")
        watched_ids = extract_ids_from_entries(user_data.get('watchedEntries', []), "watched")
        watchlist_ids = extract_ids_from_entries(user_data.get('watchlistEntries', []), "watchlist")
        all_user_ids = set(fav_ids + watched_ids + watchlist_ids) # Bunları tekrar önermeyeceğiz
        
        # --- ZEVK VEKTÖRÜNÜ YİNE DE HESAPLA (Aday çekmek için LAZIM) ---
        all_ids_to_fetch = list(all_user_ids)
        content_meta_data = get_content_from_firestore(all_ids_to_fetch)
        valid_ids = [id for id in all_ids_to_fetch if id in content_meta_data]
        if not valid_ids:
             return jsonify({"message": "Listenizdeki içerikler, öneri veritabanımızdaki içeriklerle eşleşmedi."}), 200
        vector_data = chroma_collection.get(ids=valid_ids, include=['embeddings'])
        id_to_vector_map = {id: np.array(emb) for id, emb in zip(vector_data['ids'], vector_data.get('embeddings', []))}
        all_vectors = []
        weights = []
        for id_str in valid_ids:
            if id_str in id_to_vector_map:
                all_vectors.append(id_to_vector_map[id_str])
                if id_str in fav_ids: weights.append(1.0)
                elif id_str in watched_ids: weights.append(0.75)
                elif id_str in watchlist_ids: weights.append(0.25)
        if not all_vectors: return jsonify({"error": "Geçerli içerikler için vektör bulunamadı."}), 500
        taste_vector = np.average(all_vectors, axis=0, weights=weights)
        print(f"Ağırlıklı zevk profili {len(all_vectors)} vektör üzerinden hesaplandı.")
        # --- ZEVK VEKTÖRÜ HESAPLAMA SONU ---

        final_scored_recommendations = []
        
        # --- YENİ MANTIK: TÜR FİLTRESİ VAR MI? ---
        
        # 1. ADAYLARI ÇEK
        chroma_filter = None
        if content_type_filter in ['movie', 'tv']:
            chroma_filter = {"type": content_type_filter}
        
        print(f"ChromaDB'den {CANDIDATE_POOL_SIZE} adet aday çekiliyor...")
        query_results = chroma_collection.query(
            query_embeddings=[taste_vector.tolist()],
            n_results=CANDIDATE_POOL_SIZE, 
            where=chroma_filter, 
            include=['distances'] 
        )
        candidate_ids = query_results['ids'][0]
        distances = query_results['distances'][0]
        cand_content_data = get_content_from_firestore(candidate_ids)
        
        print(f"Puanlama {len(candidate_ids)} aday için başlıyor...")
        
        # 2. ADAYLARI PUANLA (FİLTRELİ VEYA FİLTRESİZ)
        
        # Ana motor için kullanılacak KURAL setleri
        strong_signal_ids = set(fav_ids + watched_ids)
        strong_signal_data = {id: data for id, data in content_meta_data.items() if id in strong_signal_ids}
        fav_creators = set(data.get('director_or_creator', '') for data in strong_signal_data.values() if data.get('director_or_creator'))
        fav_genres = set(genre for data in strong_signal_data.values() for genre in data.get('genres', []))
        fav_actors = set(actor for data in strong_signal_data.values() for actor in data.get('actors', [])[:3])
        
        for i, cand_id in enumerate(candidate_ids):
            if cand_id in all_user_ids: continue 
            if cand_id not in cand_content_data: continue
                
            cand_content = cand_content_data[cand_id]
            
            # --- YENİ: CHATBOT "KEŞİF MODU" PUANLAMASI ---
            if genre_filters:
                # Adım 2a: Tür Filtreleme ("OR" mantığı)
                cand_genres_set = set(cand_content.get('genres', []))
                if not cand_genres_set.intersection(genre_filters):
                    continue # İstenen türlerden HİÇBİRİ yoksa atla
                
                # Adım 2b: "Keşif Puanı" Hesapla
                # (Kişisel zevk + Genel Kalite)
                content_score = normalize_content_score(distances[i], max_points=50) # Max 50
                virality_score = get_virality_score(cand_content.get('rating', 0), max_points=50) # Max 50
                
                final_score = content_score + virality_score
                
                final_scored_recommendations.append({
                    "content_id": cand_id, "type": cand_content.get('type'),
                    "title": cand_content.get('title'),
                    "poster_url": cand_content.get('poster_url'),
                    "year": cand_content.get('year'),
                    "final_score": round(final_score, 2),
                    "debug_details": {
                        "content_score (max 50)": round(content_score, 2),
                        "virality_score (max 50)": virality_score
                    }
                })

            # --- ESKİ MANTIK: "KİŞİSEL ZEVK MODU" ---
            else: 
                # Ana Motor: 30/70 Puanlama
                content_score = normalize_content_score(distances[i], max_points=30) # Max 30
                rule_score = 0
                
                # Kademeli Puanlama (Max 70)
                if cand_content.get('director_or_creator') and cand_content.get('director_or_creator') in fav_creators:
                    rule_score += 30
                cand_actors = set(cand_content.get('actors', [])[:3])
                actor_matches = len(cand_actors.intersection(fav_actors))
                if actor_matches == 1:
                    rule_score += 15
                elif actor_matches >= 2:
                    rule_score += 20
                cand_genres = set(cand_content.get('genres', []))
                genre_matches = len(cand_genres.intersection(fav_genres))
                if genre_matches == 1:
                    rule_score += 5
                elif genre_matches == 2:
                    rule_score += 10
                elif genre_matches >= 3:
                    rule_score += 15
                if cand_content.get('rating', 0) >= 8.0:
                    rule_score += 5
                
                final_score = content_score + rule_score
                
                final_scored_recommendations.append({
                    "content_id": cand_id,
                    "type": cand_content.get('type'),
                    "title": cand_content.get('title'),
                    "poster_url": cand_content.get('poster_url'),
                    "year": cand_content.get('year'),
                    "final_score": round(final_score, 2),
                    "debug_details": {
                        "content_score (max 30)": round(content_score, 2),
                        "rule_score (max 70)": rule_score
                    }
                })

        # 3. SONUÇLARI SIRALA VE FİLTRELE
        sorted_recommendations = sorted(
            final_scored_recommendations, 
            key=lambda x: x['final_score'], 
            reverse=True
        )
        
        # Aktif eşiği belirle
        active_threshold = MIN_SCORE_THRESHOLD # Varsayılan: 70.0
        if genre_filters:
            active_threshold = CHATBOT_DISCOVERY_THRESHOLD # Keşif Modu: 50.0
            print(f"Chatbot filtresi aktif. Kalite eşiği {active_threshold}'a (Keşif Modu) düşürüldü.")
        
        high_quality_recommendations = [
            rec for rec in sorted_recommendations 
            if rec['final_score'] >= active_threshold
        ]
        
        top_recommendations = high_quality_recommendations[:10]

        print(f"Toplam {len(sorted_recommendations)} adaydan (ve {len(genre_filters)} filtreden) sonra, {len(high_quality_recommendations)} tanesi {active_threshold} puan eşiğini geçti. İlk {len(top_recommendations)} tanesi döndürülüyor.")
        
        json_response = json.dumps(top_recommendations, ensure_ascii=False, indent=4)
        return Response(json_response,
                        content_type="application/json; charset=utf-8")

    except Exception as e:
        print(f"HATA: Öneri hesaplanırken bir sorun oluştu: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Sunucu hatası: Öneri hesaplanamadı."}), 500

# --- 6. SUNUCUYU ÇALIŞTIRMA ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)