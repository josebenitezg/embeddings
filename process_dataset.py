import pandas as pd
from sentence_transformers import SentenceTransformer
from ast import literal_eval
import pinecone
import os
from rich import print

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# I need to get the api key from the environment
pinecone_api = os.environ.get('PINECONE_API_KEY')
pinecone.init(api_key=pinecone_api, environment='us-west1-gcp')
    
'''
Pre processing
'''

def concatenar_lista(lista):
    lista = literal_eval(lista)
    return ' '.join(lista)

def string_to_list(lista):
    lista = literal_eval(lista)
    return lista

def process_file():
    print("Hey, [bold magenta]Processing your data[/bold magenta]!", ":vampire:", locals())
    df = pd.read_csv('assets/25k_movie_imdb.csv')

    df = df.fillna(' ')
    df['Keywords'] = df['Plot Kyeword'].apply(concatenar_lista)
    df['Stars'] = df['Top 5 Casts'].apply(concatenar_lista)
    df['Generes'] = df['Generes'].apply(string_to_list)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(0).astype('float')

    unique_generes = df['Generes'].explode().unique()

    df.drop(['Plot Kyeword', 'Top 5 Casts'], axis=1, inplace=True)

    df['text'] = df.apply(lambda x : str(x['Overview']) + ' ' + str(x['Keywords']) + ' ' + str(x['Stars']), axis=1)

    embeddings = model.encode(df['text'], batch_size=64, show_progress_bar=True)

    df['embeddings'] = embeddings.tolist()
    df['ids'] = df.index
    df['ids'] = df['ids'].astype('str')
    
    return df, unique_generes

def get_pinecone_index():
    '''
    Prepare Pinecone
    '''
    print("Oh oh, [bold magenta]Don't worry. Just checking on Pinecone[/bold magenta]!", ":book:", locals())

    dimension_embedding = len(df['embeddings'][0])

    index_name = 'movies-embeddings'
    all_index = pinecone.list_indexes()
    if index_name in all_index:
        index = pinecone.Index(index_name)
    else:
        pinecone.create_index(index_name, dimension=dimension_embedding, metric='cosine')
    index = pinecone.Index(index_name)
    
    from tqdm.auto import tqdm

    '''
    Ingest Data
    '''

    # We will use batches of 64
    batch_size = 64

    for i in tqdm(range(0, len(df), batch_size)):
        i_end = min(i + batch_size, len(df))
        # extract batch
        batch = df[i:i_end]
        ids = batch['ids']
        emb = batch['embeddings']
        metadata = batch.drop(['ids', 'embeddings', 'text', 'path'], axis=1).to_dict('records')
        # add all to upsert list
        to_upsert = list(zip(ids, emb, metadata))
        _ = index.upsert(vectors=to_upsert)

    return index

if __name__ == '__main__':
    df, unique_generes = process_file()
