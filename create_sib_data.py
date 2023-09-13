import pandas as pd
import os

def read_text(filenam):
    with open(filenam) as f:
        text_lines = f.read().splitlines()

    return text_lines


def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def get_dict_index_text(df):
    dict_title_text = {}
    N_samples = df.shape[0]
    for i in range(N_samples):
        dict_title_text[df['index_id'].iloc[i]] = df['text'].iloc[i]

    return dict_title_text

#lang = 'kik_Latn'

list_langs = set([lang.split('.')[0] for lang in os.listdir('data/raw/')])

for lang in list_langs:
    print(lang)
    dev = read_text('data/raw/'+lang+'.dev')
    devtest = read_text('data/raw/'+lang+'.devtest')

    #print(len(dev), len(devtest))

    all_texts = dev + devtest

    df_all = pd.DataFrame(all_texts, columns=['text'])

    df_all['index_id'] = list(range(df_all.shape[0]))

    dict_index_text = get_dict_index_text(df_all)

    output_dir = 'data/annotated/'+lang+'/'
    create_dir(output_dir)

    for split in ['train', 'dev', 'test']:
        df = pd.read_csv('data/eng/'+split+'.tsv', sep='\t')

        df['lang_text'] = df['index_id'].map(dict_index_text)
        df['lang_text'] = df['lang_text'].str.rstrip()
        df.columns = ['index_id', 'category', 'source_text', 'text']

        df[['index_id', 'category', 'text']].to_csv(output_dir+split+'.tsv', sep='\t', index=None)

    labels = read_text('data/eng/labels.txt')
    with open(output_dir+'labels.txt', 'w') as f:
        for l in labels:
            f.write(l+'\n')



