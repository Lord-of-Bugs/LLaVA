# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import argparse
from transformers import pipeline
import evaluate

#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sent_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def break_convos(df, col1, col2):
    '''takes in conversations column and breaks it up into separate columns for data frame'''
    ser = df['conversations']
    prompt = pd.json_normalize(ser.apply(lambda x: x[0]))['value']
    impressions = pd.json_normalize(ser.apply(lambda x: x[1]))['value']

    df[col1] = prompt
    df[col2] = impressions

    df = df.drop(axis=1, columns=['conversations'])

    return df

def read_image(image_path):
    # Read the image
    img = mpimg.imread(image_path)

    # Display the image
    plt.imshow(img)
    plt.axis('off')  # Turn off axis numbers
    plt.show()

def select_random_img():
    idx = np.random.randint(merged_df.shape[0])
    row = merged_df.iloc[idx]

    read_image(row['image'])
    print(f"id: {row['id']}\nradiologist report: {row['radiologist_report']}\nllava_prompt: {row['llava_prompt']}\nimpression: {row['llava_impression']}")


def read_and_process(test_questions_path, test_inference_path, train_data_path):
    test_questions = pd.read_json(test_questions_path, lines=True)
    test_inference = pd.read_json(test_inference_path, lines=True)
    train_df = pd.read_json(train_data_path).drop(axis=1, columns='image')
    train_df = break_convos(train_df, 'prompt', 'radiologist_report')
    csv_t2020_df = pd.read_csv('/home/llm-hackathon/Downloads/fixed-data-csv/all-reports-available-xrays-through2020-final.csv').drop(axis=1, columns='Unnamed: 0')


    test_inference[['phonetic_id', 'id']] = test_inference['question_id'].str.split('_', expand=True)
    test_questions[['phonetic_id', 'id']] = test_questions['question_id'].str.split('_', expand=True)
    train_df[['phonetic_id', 'id']] = train_df['id'].str.split('_', expand=True)

    test_inference['id'] = test_inference['id'].astype(int)
    train_df['id'] = train_df['id'].astype(int)

    test_inference.rename(columns={'text': 'llava_report'}, inplace=True)
    test_questions.rename(columns={'text': 'prompt'}, inplace=True)

    test_inference = test_inference[['id', 'phonetic_id', 'prompt', 'llava_report']]
    test_questions = test_questions[['id', 'phonetic_id', 'prompt']]

    # split information into separate columns for readability
    # train df 
    train_df['author'] = train_df['prompt'].str.split('\n', expand=True)[0].str.split('AUTHOR: ', expand=True)[1].str.strip()
    train_df['clinical_history'] = train_df['prompt'].str.split('CLINICAL HISTORY:', expand=True)[1].str.split('\nBased on AUTHOR and', expand=True)[0].str.strip()
    train_df[['findings', 'impression']] = train_df['radiologist_report'].str.split('IMPRESSION:', expand=True)[[0, 1]]
    train_df['findings'] = train_df['findings'].str.split('FINDINGS:', expand=True)[1].str.strip()

    # test inference
    test_inference['author'] = test_inference['prompt'].str.split('\n', expand=True)[0].str.split('AUTHOR: ', expand=True)[1].str.strip()
    test_inference['clinical_history'] = test_inference['prompt'].str.split('CLINICAL HISTORY:', expand=True)[1].str.split('\nBased on AUTHOR and', expand=True)[0].str.strip()
    test_inference[['findings', 'impression']] = test_inference['llava_report'].str.split('IMPRESSION:', expand=True)[[0, 1]]
    test_inference['findings'] = test_inference['findings'].str.split('FINDINGS:', expand=True)[1].str.strip()

    # test questions has no new information (besides the question asked), which hasn't been changed all that much, so 
    # we won't focus on this data frame for now
    # print question asked
    print('Test Set Prompt:')
    print(test_questions['prompt'].str.split('\n', expand=True)[2].unique()[0])

    train_df = train_df[['id', 'phonetic_id', 'author', 'clinical_history', 'findings', 'impression']]
    test_inference = test_inference[['id', 'phonetic_id', 'author', 'clinical_history', 'findings', 'impression', 'llava_report']]

    train_df.rename(columns={'findings': 'radiologist_findings', 'impression': 'radiologist_impression'}, inplace=True)
    test_inference.rename(columns={'findings': 'llava_findings', 'impression': 'llava_impression'}, inplace=True)

    # 'DocumentTitle', 'DocumentDate' are columns we can use from csv_t2020_df
    test_inference = test_inference.merge(csv_t2020_df[['AccessionId', 'answer']], left_on='id', right_on='AccessionId').drop(axis=1, columns='AccessionId') 
    test_inference[['radiologist_findings', 'radiologist_impression']] = test_inference['answer'].str.split('IMPRESSION:', expand=True)[[0, 1]]
    test_inference['radiologist_findings'] = test_inference['radiologist_findings'].str.split('FINDINGS:', expand=True)[1].str.strip()
    test_inference.rename(columns={'answer': 'radiologist_report'}, inplace=True)#test_inference.drop(axis=1, columns='answer', inplace=True)

    # replace None rows so we don't get an error
    test_inference['radiologist_impression'] = test_inference['radiologist_impression'].fillna(' ')
    test_inference['llava_impression'] = test_inference['llava_impression'].fillna(' ')

    string_cols = ['phonetic_id', 'author', 'clinical_history', 'llava_findings', 'llava_impression', 'llava_report', \
    'radiologist_report', \
    'radiologist_findings', \
    'radiologist_impression']

    for col in string_cols:
        test_inference[col] = test_inference[col].str.strip()
    
    return test_inference

def get_sim_scores(col1, col2):
    '''takes in two columns which each row in each column a sentence that corresponds with each other. 
    returns a similarity score series the same length as inputs'''
    
    embeddings1 = sent_model.encode(col1)
    embeddings2 = sent_model.encode(col2)
    
    return embeddings1, embeddings2, np.diag(embeddings1 @ embeddings2.T)

def sample_by_bin(df, col, bin_index):
    '''
    col -- name of column to get similarities from
    bin_index -- a number from 1 to 11 (probably)'''
    sim_scores = df[col]
    bin_indices = np.digitize(sim_scores, bins=np.histogram_bin_edges(sim_scores))
    bools = bin_indices == bin_index
    filtered = sim_scores[bools]
    df_subset = df[bools]
    
    i = np.random.randint(0, high=len(filtered))
    return df_subset.iloc[i]

def print_row_nicely(row):
    """
    This function takes a row of data (as a dictionary) and prints it out in a formatted manner.
    
    Parameters:
    row (dict): A dictionary representing a row of data, where keys are column names.
    """
    read_image('/data/UCSD_cxr/jpg/' + row['phonetic_id'] + '_' + str(row['id']) + '.jpg')
    print("Row Data Overview:")
    print("-" * 50)  # Print a divider for better readability
    for key, value in row.items():
        if key == 'llava_report':
            pass
        elif key =='radiologist_report':
            pass
        # Formatting each key-value pair nicely
        else:
            print(f"{key}:".ljust(25) + f"{value}")
    print("-" * 50)  # End with a divider for a clean look

sampled_row = sample_by_bin(test_inference, 'similarity', 2)
print_row_nicely(sampled_row)

def calculate_text_lengths(test_inference):
    see_impression_bools = test_inference['llava_findings'].str.lower().str.contains('see impression') | test_inference['llava_findings'].str.lower().str.contains('unknown') | (test_inference['llava_findings'] == '')
    see_impression = test_inference['llava_findings'].str.lower()[see_impression_bools]
    see_impression_lens = see_impression[see_impression_bools].str.len()

    test_inference['is_si_l'] = see_impression_bools 
    test_inference['is_si_r'] = test_inference['radiologist_findings'].str.lower().str.contains('see impression') | test_inference['radiologist_findings'].str.lower().str.contains('unknown') | (test_inference['radiologist_findings'] == '')

    # %%
    llava_impression_lengths = test_inference['llava_impression'].str.len()
    llava_findings_lengths = test_inference['llava_findings'].str.len()
    radiologist_impression_lengths = test_inference['radiologist_impression'].str.len()
    radiologist_findings_lengths = test_inference['radiologist_findings'].str.len()

    test_inference['lf_len'] = llava_findings_lengths
    test_inference['li_len'] = llava_impression_lengths
    test_inference['rf_len'] = radiologist_findings_lengths
    test_inference['ri_len'] = radiologist_impression_lengths


def plot_lengths(threshold=1000):
    # TODO: save these plots to the output directory
    llava_findings_lengths[llava_findings_lengths < threshold].plot(kind='hist') # excluding the nonsensical outliers
    plt.title('Llava Findings Section Lengths')
    plt.axvline(np.median(llava_findings_lengths), color='red')


    radiologist_findings_lengths.plot(kind='hist')
    plt.title('Radiologist Findings Section Lengths')
    plt.axvline(np.median(radiologist_findings_lengths), color='red')

    llava_impression_lengths[llava_impression_lengths < threshold].plot(kind='hist')
    plt.title('Llava Impression Section Lengths')
    plt.axvline(np.median(llava_impression_lengths), color='red')


    radiologist_impression_lengths.plot(kind='hist')
    plt.title('Radiologist Impression Section Lengths')
    plt.axvline(np.median(radiologist_impression_lengths), color='red')

def plot_sim_scores(title='Report Similarity', col='similarity'):
    '''creates histogram plot of similarity scores from the specified column'''
    # TODO: rewrite this function so that it can be used for 
    # TODO: need to save to save the data
    ax = test_inference[col].hist()
    plt.title(title) 
    plt.axvline(test_inference[col].median(), color='red')
    plt.xlabel('Similarity Score (0-1)')
    plt.ylabel('Counts')


def sim_score_by_author(author, sim_col='similarity'):
    '''creates a plot of that author's sim scores'''
    filtered = test_inference[test_inference['author'] == author]
    filtered[sim_col].plot(kind='hist')
    plt.title(f'{author}')
    plt.xlabel('similarity scores')

def get_top_authors(test_inference):
    author_medians = dict()
    top_authors = set(test_inference['author'].value_counts()[0:10].index)
    for author in top_authors:
        author_medians[author] = np.median(test_inference[test_inference['author'] == author]['similarity'])
    return author_medians


def plot_by_author(top_authors, author_col='author', sim_col='similarity', title=''):
    '''plots a bar plot with the median similarity for each author'''
    test_inference.groupby(author_col)[sim_col].median().loc[list(top_authors)].sort_values().plot(kind='bar')
    plt.title(title)


def create_labels(test_inference, candidate_labels, col='radiologist_report'):
    '''create a classifier with facebook's bart
    
    130 seconds per report runtime'''
    
    classifier = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli", multilabel=True)
    
    sequences = test_inference[col]
    
    classsifier(sequences, candidate_labels)
    
    # TODO COMPLETE THIS FUNCTION TO INCORPORATE THE DATA BACK TO DATA FRAME AND USE FOR SOMETHING 


def add_rouge_scores(test_inference, reference_col='radiologist_report', candidate_col='llava_report'):

    rouge = evaluate.load('rouge')

    test_inference[['rouge1', 'rouge2', 'rougeL', 'rougeLsum']] = test_inference.apply(lambda x: rouge.compute(predictions=[x[candidate_col]], references=[x[reference_col]]), 
                            axis=1, result_type='expand')



def main(input_csv, output_csv, plot_out):
    # can add argument for sentence transformer
    sent_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    
    test_questions_path = '../data/01-29-24/test_questions_finding_impression.jsonl' 
    test_inference_path = '../data/01-29-24/test_inference.jsonl'
    train_data_path = '../data/01-29-24/train_patient_finding_impression.json'
    
    test_inference = read_and_process(test_questions_path, test_inference_path, train_data_path)
    
    llava_embeddings, radiologist_embeddings, test_inference['similarity'] = get_sim_scores(test_inference['llava_report'], test_inference['radiologist_report'])
    
    test_inference = test_inference[['id',
                                    'phonetic_id',
                                    'author',
                                    'clinical_history',
                                    'llava_findings',
                                    'llava_impression',
                                    'llava_report',
                                    'radiologist_report',
                                    'radiologist_findings',
                                    'radiologist_impression',
                                    'similarity']]
    
    impression_embeddings, radiologist_impression_embeddings, test_inference['impression_similarity'] = get_sim_scores(test_inference['llava_impression'], test_inference['radiologist_impression'])
    llava_findings_embeddings, radiologist_findings_embeddings, test_inference['findings_similarity'] = get_sim_scores(test_inference['llava_findings'], test_inference['radiologist_findings'])
    
    candidate_labels = [
                        'No Abnormality',
                        'Aortic enlargement',
                        'Atelectasis',
                        'Calcification',
                        'Cardiomegaly',
                        'Consolidation',
                        'ILD',
                        'Infiltration',
                        'Lung Opacity',
                        'Nodule/Mass',
                        'Other lesion',
                        'Pleural effusion',
                        'Pleural thickening',
                        'Pneumothorax',
                        'Pulmonary fibrosis']
    
    # TODO: edit and add in functions 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance and generate plots.")
    parser.add_argument("input_csv", help="Path to the input CSV file containing unseen data and ground truth.")
    parser.add_argument("output_csv", help="Path where the output CSV file will be saved.")
    
    # You can add more arguments here as needed
    
    args = parser.parse_args()
    
    main(args.input_csv, args.output_csv)

