# -*- coding: utf-8 -*-
from models import DeepTilingModels
from LexRank import degree_centrality_scores
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
import shutil
import sys
import os
import segment as seg
import pandas as pd

def main(args):

    out_directory = args.out_directory
    
    deeptiling = DeepTilingModels.DeepTiling(args.encoder)
    
    search = deeptiling.encoder.encode([input('Write below your query:\n\n')])
    
    if not os.path.exists(os.path.join(out_directory, 'segments')):
        seg.main(args)
    
    embeddings_df = {}
    
    segments_df = {}
    
    for root, directory, files in os.walk(os.path.join(out_directory, 'segments')):
        for file in files:
            if file.startswith('paths_cach'):
                pass
            else:
                segment = []
                
                with open(os.path.join(out_directory, 'segments', file), 'r') as f:
                    for line in f:
                        segment.append(line)
                        
                embeddings = np.load(os.path.join(out_directory, 'embeddings', file+'.npy'))
                
                segments_df[file] = segment
                
                if args.number_top_sentences>=len(segment):
                    embeddings_df[file] = np.mean(embeddings, axis=0)
                    
                else:
                    
                    
                    #Compute the pair-wise cosine similarities
                    cos_scores = util.pytorch_cos_sim(embeddings, embeddings).numpy()
                    
                    #Compute the centrality for each sentence
                    centrality_scores = degree_centrality_scores(cos_scores, threshold=None)
                    
                    #We argsort so that the first element is the sentence with the highest score
                    most_central_sentence_indices = np.argsort(-centrality_scores)
                    
                    top_index = 1
                    top_embeddings = []
                    for idx in most_central_sentence_indices[:args.number_top_sentences]:
                        
                        top_sentence = segment[idx]
                        
                        top_embeddings.append(embeddings[idx])
                    
                    embeddings_df[file] = np.mean(top_embeddings, axis=0)
    
    embeddings_df = pd.DataFrame(embeddings_df)
    
    similarity_scores = util.pytorch_cos_sim(embeddings_df.T.values, search).numpy()
    
    result = embeddings_df.columns[np.argmax(similarity_scores)]
    
    print('The most relevant segment is {}'.format(result))
    
    if args.verbose:
        print('Returning the entire segment below:\n')
        
        print(' '.join(segments_df[result]))
    
    return result    
        
                    
if __name__ == '__main__':
    
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
    
    parser = MyParser(
            description = 'Run segmentation and summarization with parameters defined in the relative json file\
                or as passed in the command line')
    
    parser.add_argument('--data_directory', '-data', type=str,
                        help='directory containing the data to be segmented')
    
    parser.add_argument('--config_file', '-cfg', default='parameters.json', type=str, 
                        help='Configuration file defining the hyperparameters and options to be used in training.')
    
    parser.add_argument('--out_directory', '-od', default='results', type=str,
                        help='the directory where to store the segmented texts')
    
    
    parser.add_argument('--number_top_sentences', '-nt', type=int,
                        default=1, help='number of sentences to extract per segment as summary')
    
    parser.add_argument('--window_value', '-wd', 
                        type=int,
                        default=None, 
                        help='Window value for the TextTiling algorithm, if not specified the programme will assume that the optimal value is stored in best_parameters.json file, previously obtained by running fit.py')
    
    parser.add_argument('--threshold_multiplier', '-th',
                        type=float,
                        default=None,
                        help='Threshold multiplier for the TextTiling algorithm without known number of segments, if not specified the programme will assume that the optimal value is stored in best_parameters.json file, previously obtained by running fit.py')
    
    parser.add_argument('--number_of_segments', '-ns',
                        type=int,
                        nargs = '+',
                        default=None,
                        help='Number of segments to be returned (if known). Default is when number of segments are not None, otherwise the algorithm returns the n number of segments with higher depth score, as specified by this parameter')
    
    parser.add_argument('--encoder', '-enc', type=str,
                        default=None, help='sentence encoder to be used (all sentence encoders from sentence_transformers library are supported)')
    
    parser.add_argument('--Concatenate', '-cat', type=bool,
                        default=None, help='whether to concatenate the input files or to segment them individually')
    
    parser.add_argument('--verbose', '-vb', type=bool, default=True, help='Whether to print messages during running.')
    
    args = parser.parse_args()
    
    main(args)