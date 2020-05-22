import pandas as pd
import numpy as np
import simplejson

def duplicated_search(list):
    for i in range(len(list)):
        for j in range(i+1, len(list)):
            if list[i]==list[j]:
                print('duplicated found:')
                print('\t',list[j])

# read all data json
wines = pd.read_json('data_wines.json', orient='records', encoding='utf-8')
wineries = pd.read_json('data_wineries.json', orient='records', encoding='utf-8')

print('\nExample Wines')
print(wines.head(5))
print('\nExample Wineries')
print(wineries.head(5))

# will select only k wines in our data and corresponding wineries
k = 100

############# WINES ##################################################################
first_k = wines.iloc[0:k]

first_k = first_k.rename(columns={'id_site': 'id'})
first_k = first_k.drop(columns=['last_modified','sub_category'])

# now write wines to a file
wines_json_file = open('wines.json', 'w', encoding='utf-8')
# magic happens here to make it pretty-printed
wines_json_file.write(simplejson.dumps(simplejson.loads(first_k.to_json(orient='records')), indent=4, sort_keys=True, ensure_ascii=False))
wines_json_file.close()
print('\nWines printed to json file!')

############# WINERIES ##################################################################
winery_ids_for_first_k = first_k['winery_id']

# find duplicated wineries
duplicated_search([id for id in winery_ids_for_first_k])

unique_winery_ids = np.unique(winery_ids_for_first_k)

wineries_for_first_k_wines = wineries.loc[wineries['id_site'].isin(unique_winery_ids)]
wineries_for_first_k_wines = wineries_for_first_k_wines.rename(columns={'id_site': 'id'})

# now write wineries to a file
wineries_json_file = open('wineries.json', 'w', encoding='utf-8')
# magic happens here to make it pretty-printed
wineries_json_file.write(simplejson.dumps(simplejson.loads(wineries_for_first_k_wines.to_json(orient='records')), indent=4, sort_keys=True, ensure_ascii=False))
wineries_json_file.close()
print('\nWineries printed to json file!')
