from pathlib import Path
import requests 
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import json
import argparse
from tqdm.auto import tqdm
tqdm.pandas()


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--download', action='store_true',
                        help='Download single amino acid reference properties')
    parser.add_argument('-p', '--props', action='store_true',
                        help='Calculate kmer properties as sum of individual amino acids')

    return parser.parse_args()


def get_code():
    
    '''aa name to three letter and one letter'''

    url = 'https://www.anaspec.com/html/amino_acid_codes.html'
    r = requests.get(url)
    assert r.status_code == 200, 'invalid request'
    soup = BeautifulSoup(r.content, 'lxml')
    
    aa_table = soup.find_all('table')[1]
    
    res = []
    for tr in aa_table.find_all('tr'):
        aa_data = []
        for td in tr.find_all('td'):
             aa_data.append(td.text.strip())
                
        res.append(aa_data)
        
    del res[0] 
        
    df = pd.DataFrame(res, columns=['name', 'aa_3', 'aa_1'])
    
    return df


def get_mass():

    url = 'http://education.expasy.org/student_projects/isotopident/htdocs/aa-list.html'
    r = requests.get(url)
    assert r.status_code == 200, 'invalid request'
    soup = BeautifulSoup(r.content, 'lxml')
    
    # get data from table
    aa_table = soup.find('table')
    res = []
    for tr in aa_table.find_all('tr'):
        aa_data = []
        for td in tr.find_all('td'):
            aa_data.append(td.text)
            
        res.append(aa_data)
        
    # drop first blank
    del res[0]
   
    df = pd.DataFrame(res, columns=['aa_1', 'aa_3', 'chem_struct', 'monoiso', 'mass'])
    
    return df


def get_pi():

    '''isoelectric point'''
    
    url = 'http://www.chem.ucalgary.ca/courses/351/Carey5th/Ch27/ch27-1-4-2.html'
    r = requests.get(url)
    assert r.status_code == 200, 'invalid request'
    soup = BeautifulSoup(r.content, 'lxml')
    
    # second table on page
    aa_table = soup.find_all('table')[1]
    
    res = []
    for tr in aa_table.find_all('tr'):
        aa_data = []
        for td in tr.find_all('td'):
            aa_data.append(td.text.strip())
            
        res.append(aa_data)
        
    # delete header and blank end
    del res[0]
    del res[-1]
    
    df = pd.DataFrame(res, columns=['aa', 'pka_1', 'pka_2', 'pka_3', 'pI'])
    df['aa'] = df['aa'].apply(lambda x: x.title())  # capitalize Acid in aspartic/glutamic acid to merge later

    # pI saving invisible chars
    # keep only numeric
    # pat = re.compile(r'\d+\.\d+')
    # df['pI'] = df['pI'].apply(lambda x: pat.match(x).group(0))
            
    return df


def get_volume():
    
    url = 'http://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/abbreviation.html'
    r = requests.get(url)
    assert r.status_code == 200, 'invalid request'
    soup = BeautifulSoup(r.content, 'lxml')
    
    aa_table = soup.find('table')
    
    res = []
    for tr in aa_table.find_all('tr'):
        aa_data = []
        for td in tr.find_all('td'):
            aa_data.append(td.text)
            
        res.append(aa_data)
        
    # blank header
    del res[0] 
        
    df = pd.DataFrame(res, columns=['aa', 'aa_3', 'aa_1', 'mass', 'n_atoms', 'volume', 'hydropathy'])
    
    # drop asx and glx
    df = df[~df['aa'].str.contains('or')]
            
    return df


def get_hydrophobicity():
    
    # will not work with default request headers
    headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}
    r = requests.get('https://www.sigmaaldrich.com/life-science/metabolomics/learning-center/amino-acid-reference-chart.html', 
                     headers=headers)
    assert r.status_code == 200, 'invalid request'
    soup = BeautifulSoup(r.content, 'lxml')
    
    to_skip = ['At', 'Hydrophobic', 'Neutral', 'Hydrophilic', 'values']

    aa_names = []
    aa_hydro = []
    
    aa_table = soup.find_all('table')[1]

    # save second 
    for tr in aa_table.find_all('tr'):
        for td in tr.find_all('td'):
            # skip the headers and footnotes
            if any(x in td.text.strip() for x in to_skip) or len(td.text.strip()) == 0:
                continue

            txt = td.text.strip()

            # save aa names
            if txt.isalpha():
                aa_names.append(txt)

            # save hydrophobicities
            # account for edge case for pro note in parenthesis
            else:
                aa_hydro.append(txt.split('(')[0])

    # remove first col, manually select due to ragged table
    aa_names = aa_names[1:28:2] + aa_names[30::2] + aa_names[-1:]
    aa_hydro = aa_hydro[1:28:2] + aa_hydro[30::2] + aa_hydro[-1:]
    
    df = pd.DataFrame({'aa_3' : aa_names,
                       'hydrophobicity' : aa_hydro})

    return df


def get_vdw_volume():
    
    url = 'https://en.wikipedia.org/wiki/Proteinogenic_amino_acid'
    r = requests.get(url)
    assert r.status_code == 200, 'invalid request'
    soup = BeautifulSoup(r.content, 'lxml')
    
    aa_table = soup.find_all('table')[1]
    res = []
    for tr in aa_table.find_all('tr'):
        aa_data = []
        for td in tr.find_all('td'):
            aa_data.append(td.text.strip())

        res.append(aa_data)

    # empty header
    del res[0] 
        
    df = pd.DataFrame(res, columns=['aa_1', 'aa_3', 'side_chain', 'hydro', 'pka', 
                                    'polar', 'ph', 'small', 'tiny', 'ali_aro', 'vdw_vol'])
    
    df = df.query("vdw_vol != '?'")
    
    return df[['aa_1', 'aa_3', 'vdw_vol']]


def get_polarity():
    
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    polarity = [7, 8, 18, 17, 4, 9, 13, 2, 15, 1, 5, 16, 11.5, 14, 19, 12, 11, 3, 6, 10]
    
    df = pd.DataFrame({'aa_1' : list(aa),
                       'polarity' : polarity})
    
    return df


def combine():

    code = get_code()
    mass = get_mass()
    vol = get_volume()
    vdw_vol = get_vdw_volume()
    polarity = get_polarity()
    hydrophobicity = get_hydrophobicity()
    pi = get_pi()

    df = mass.merge(vol, on='aa_1')
    df = df.merge(vdw_vol, on='aa_1')
    df = df.merge(polarity, on='aa_1')
    df = df.merge(hydrophobicity, on='aa_3')
    df = df.merge(code, on='aa_1')
    df = df.merge(pi, left_on='name', right_on='aa')

    df = df[['aa_1', 'mass_x', 'volume', 'vdw_vol', 'polarity', 'hydrophobicity', 'pI']]
    df = df.rename(columns={'mass_x' : 'mass'})

    df = df.apply(pd.to_numeric, errors='ignore')
    df.to_csv('inputs/aa_props.csv', index=False)


def calc_props():

    def kmer_prop(kmer, aa2idx, prop):

        res = 0
        for aa in kmer:
            idx = aa2idx.query("aa_1 == @aa").index.values[0]
            res += prop.values[idx]

        return res

    kmers_dict = json.loads(Path('./preprocessed/word2idx.json').read_bytes())
    kmers = list(kmers_dict.keys())[:-1]  # ignore <unk>
    df = pd.DataFrame({'kmer' : kmers})

    aa_props = pd.read_csv('inputs/aa_props.csv')
    aa2idx = aa_props[['aa_1']]

    df['mass'] = df['kmer'].progress_apply(lambda x: kmer_prop(x, aa2idx, aa_props['mass']))
    df['charge'] = df['kmer'].progress_apply(lambda x: kmer_prop(x, aa2idx, aa_props['pI']))  # paper used pI for charge
    df['vol'] = df['kmer'].progress_apply(lambda x: kmer_prop(x, aa2idx, aa_props['volume']))
    df['vdw_vol'] = df['kmer'].progress_apply(lambda x: kmer_prop(x, aa2idx, aa_props['vdw_vol']))
    df['hydrophobicity'] = df['kmer'].progress_apply(lambda x: kmer_prop(x, aa2idx, aa_props['hydrophobicity']))
    df['polarity'] = df['kmer'].progress_apply(lambda x: kmer_prop(x, aa2idx, aa_props['polarity']))

    df.to_csv('inputs/kmers_props_calc.csv', index=False)

if __name__ == '__main__':

    args = parse_args()

    if args.download:
        combine()

    if args.props:
        calc_props()
