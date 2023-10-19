import pandas as pd
from sentence_transformers import SentenceTransformer
from process_dataset import get_pinecone_index

preprocessing = False
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
index = get_pinecone_index()

def search(query, genre, rating, top_k):
  query_vector = model.encode(query).tolist()

  if rating:
    filter_rating = rating
  else:
    filter_rating = 0

  if genre:
    conditions = {
        'Generes': {
            "$in": [genre]
        },
        'Rating': {
            "$gte": filter_rating
        }
    }
  else:
    conditions = {
        'Rating': {
            "$gte": filter_rating
        }
    }


  responses = index.query(
      query_vector,
      top_k=top_k,
      include_metadata=True,
      filter=conditions)

  responses_data = []
  for response in responses['matches']:
    responses_data.append(
        {
            'Title': response['metadata']['movie title'],
            'Overview': response['metadata']['Overview'],
            'Director': response['metadata']['Director'],
            'Generes': response['metadata']['Generes'],
            'Year': response['metadata']['year'],
            'Rating': response['metadata']['Rating'],
            'Score': response['score'],

        }
    )

  df = pd.DataFrame(responses_data)
  return df

