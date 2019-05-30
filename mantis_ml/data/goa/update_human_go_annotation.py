import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import datetime
from multiprocessing.dummy import Pool as ThreadPool 
import requests, sys



goa_human_gaf_gz = None

try:
    now = datetime.datetime.now()
    cur_date_str = now.strftime("%d_%m_%Y")
    print("Current date:", cur_date_str)

    print(">> Downloading updated human GOA gaf annotation file...")
    goa_human_gaf_gz = cur_date_str + '.' + 'goa_human.gaf.gz'
    urlretrieve('ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz', goa_human_gaf_gz)
    
    goa_human_df = pd.read_csv(goa_human_gaf_gz, compression='gzip', sep='\t', comment='!', header=None)
    goa_human_df = goa_human_df.iloc[:, [2,4]]
    goa_human_df.columns = ['Gene_Name', 'GO_ID']
    
except:
    print("\n[Exception]: Could not download goa_human.gaf.gz from EBI GOA FTP.\n>> Using local file instead...")
    goa_human_gaf_gz = '12_12_2018.goa_human.gaf.local_copy.gz'
    goa_human_df = pd.read_csv(goa_human_gaf_gz, compression='gzip', sep='\t', comment='!', header=None)
    goa_human_df = goa_human_df.iloc[:, [2,4]]
    goa_human_df.columns = ['Gene_Name', 'GO_ID']
        

all_human_go_ids = goa_human_df.GO_ID.unique()
np.savetxt("all_human_go_ids.txt", all_human_go_ids, fmt='%s')
print('.. goa_human.gaf.gz download complete.')

# Using Biomart
print(">>\n Retrieving GO terms from Biomart...")
num_threads = 100
pool = ThreadPool(num_threads) 

server = "https://rest.ensembl.org"
ext_prefix = "/ontology/id/"

go_id_terms_dict = {}
global_cnt = 0

def get_go_term_by_id(id):

    global global_cnt
    global_cnt += 1

    if global_cnt % 100 == 0:
        print(global_cnt)

    try:
        ext = ext_prefix + id + "?content-type=application/json"
        r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})

        if not r.ok:
            r.raise_for_status()
            return ''

        decoded = r.json()
        go_term = repr(decoded['name'])
        go_term = go_term.replace("'", "")
        go_term = go_term.replace("\"", "")
        # print(id, go_term)

    except:
        go_term = ''
        print('[Warning] Could not fetch GO term for ID:', id)
    
    go_id_terms_dict[id] = go_term
    result = id + '||' + go_term
    #print(result)


_ = pool.map(get_go_term_by_id, all_human_go_ids)



missing_go_ids = np.array([k for k,v in go_id_terms_dict.items() if v == ''])

print('\nFailed GO id->term queries:', len(missing_go_ids))
print(missing_go_ids)


## Retrieve missing GO terms from QuickGO
num_threads = 5
pool = ThreadPool(num_threads) 

server = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/"

def get_go_term_by_id(id):
    
    safe_id = id.replace(':', '%3A')
    
    try:
        go_term = ''

        req_url = server + safe_id
        r = requests.get(req_url, headers={ "Content-Type" : "application/json"})

        decoded = r.json()
        
        is_obsolete = decoded['results'][0]['isObsolete']
        if is_obsolete:
            go_term = 'is_obsolete'
        else:    
            go_term = decoded['results'][0]['name']
    
        go_id_terms_dict[id] = go_term
        print(id, go_term)
        
    except Exception as e:
        print("[Error]: ", e)
        print('[Warning]: Could not fetch GO term for ID:', id)
        go_id_terms_dict[id] = ''
        
_ = pool.map(get_go_term_by_id, missing_go_ids)


go_terms_df = pd.DataFrame.from_dict(go_id_terms_dict, orient='index')
go_terms_df.reset_index(inplace=True)
go_terms_df.columns = ['GO_ID', 'GO_term']
print(go_terms_df.head())


human_go_df = pd.merge(goa_human_df, go_terms_df, how='left', left_on = 'GO_ID', right_on='GO_ID')
# convert all GO terms to lowercase
human_go_df['GO_term'] = human_go_df['GO_term'].str.lower()
print(human_go_df.shape)
print(human_go_df.head())


human_go_df.to_csv('full_human_go_terms.txt', sep='\t', index=None)
