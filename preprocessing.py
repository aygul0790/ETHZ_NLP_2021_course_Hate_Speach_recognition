import pathlib
import pandas as pd
import re
import spacy
import multiprocessing as mp
import time
import os 

en_model = spacy.load("en_core_web_lg")
stop_words = en_model.Defaults.stop_words

def preprocessor(string):
    # lowercase everything, remove non-alphabetic characters
    string = re.compile('[^a-z ]').sub('', string.lower()).split()

    # remove stop words, lemmatize and embed each element to a vector
    string = [word for word in string if not word in stop_words]

    # transfer to spacy nlp object
    string = ' '.join(string)
    string = en_model(string)

    # lemmatize and embed each element to a vector
    output = [en_model(word.lemma_).vector for word in string]

    return output

def parallel_process(df, col):
    cores = 20

    start = time.time()

    with mp.Pool(cores) as pool:
        mp_result = pool.map(preprocessor, df[col].to_list())
        pool.close()
        pool.join()

    end = time.time()
    print('This step took ' + str((end - start) / 60) + ' minutes!')

    return mp_result

def main():
    # start with the baseline dataset to train on
    basedir = "/cluster/home/gpatoulidis/data_code"
    outdir = "/cluster/scratch/gpatoulidis/results"
    pathlib.Path(os.path.join(outdir, 'JIGSAW')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(outdir, 'HASOC')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(outdir, 'T_DAVIDSON')).mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(os.path.join(basedir, 'JIGSAW/all_data.csv'), index_col='id')
    base = base[['split', 'comment_text', 'toxicity']]
    base.dropna(inplace=True)

    base_result = parallel_process(base, 'comment_text')
    base = base[['split', 'toxicity']] # drop out the 'comment_text' column to prevent memory / performance issues with the dataframe
    base['processed_comment_text'] = base_result

    base.to_pickle(os.path.join(outdir, 'JIGSAW/preprocessed_all_data.pkl'))

    # continue with the HASOC dataset
    hasoc = pd.read_csv(os.path.join(basedir, 'HASOC/english_dataset.tsv'), sep='\t', index_col='text_id')
    hasoc_test = pd.read_csv(os.path.join(basedir, 'HASOC/hasoc2019_en_test-2919.tsv'), sep='\t', index_col='text_id')
    hasoc.dropna(inplace=True)
    hasoc_test.dropna(inplace=True)

    hasoc_result = parallel_process(hasoc, 'text')
    hasoc_test_result = parallel_process(hasoc_test, 'text')
    hasoc['processed_text'] = hasoc_result
    hasoc_test['processed_text'] = hasoc_test_result
    hasoc.to_pickle(os.path.join(outdir, "HASOC/preprocessed_english_dataset.pkl"))
    hasoc_test.to_pickle(os.path.join(outdir, "HASOC/preprocessed_hasoc2019_en_test-2919.pkl"))

    # continue with the t-davidson dataset
    # be aware: the unnamed index column does not increase by 1 after each row
    t_davidson = pd.read_csv(os.path.join(basedir, 'T_DAVIDSON/t_davidson.csv'), index_col='Unnamed: 0')
    t_davidson.dropna(inplace=True)

    t_davidson_result = parallel_process(t_davidson, 'tweet')
    t_davidson['processed_tweet'] = t_davidson_result
    t_davidson.to_pickle(os.path.join(outdir, "T_DAVIDSON/preprocessed_t_davidson.pkl"))

if __name__ == main():
    main()
