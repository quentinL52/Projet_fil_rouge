# retourne le score de similarité entre les offres d'emploi et le candidat.
from sqlalchemy import create_engine, text
from config import get_engine

engine = get_engine()

def find_similar_job_offers(limit=10):
    query = """
    WITH candidate_embedding AS (
        SELECT embedding
        FROM public.candidat
        ORDER BY name
        LIMIT 1
    )
    SELECT 
        df.entreprise, 
        df.publication, 
        df.poste, 
        df.contrat, 
        df.ville, 
        df.lien, 
        df.description_poste,
        1 - (df.embedding <=> candidate_embedding.embedding) AS similarity
    FROM public.data_fil_rouge df, candidate_embedding
    ORDER BY similarity DESC
    LIMIT :limit;
    """
    params = {'limit': limit}

    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        results = result.fetchall()
    
    return results

# Exemple d'utilisation
similar_offers = find_similar_job_offers(limit=10)
for offer in similar_offers:
    print(f"Entreprise: {offer[0]}, Poste: {offer[2]}, Similarity: {offer[7]}")
