# NCC: Neural Code Comprehension
# https://github.com/spcl/ncc
# Copyright 2018 ETH Zurich
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the follo
# wing conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
"""Helper variables and functions for NCC task training"""

import os
import pickle
import re
import struct
import zipfile
from collections import defaultdict

import wget
from absl import flags

from . import inst2vec_preprocess as i2v_prep
from . import rgx_utils as rgx

# Embedding and vocabulary file paths
flags.DEFINE_string('embeddings_file', 'published_results/emb.p',
                    'Path to the embeddings file')
flags.DEFINE_string('vocabulary_dir', 'published_results/vocabulary',
                    'Path to the vocabulary folder associated with those embeddings')

FLAGS = flags.FLAGS


########################################################################################################################
# Downloading data sets
########################################################################################################################
def download_and_unzip(url, dataset_name, data_folder):
    """
    Download and unzip data set folder from url
    :param url: from which to download
    :param dataset_name: name of data set (for printing)
    :param data_folder: folder in which to put the downloaded data
    """
    print('Downloading', dataset_name, 'data set...')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    data_zip = wget.download(url, out=data_folder)
    print('\tunzipping...')
    zip_ = zipfile.ZipFile(data_zip, 'r')
    assert os.path.isdir(data_folder), data_folder
    zip_.extractall(data_folder)
    zip_.close()
    print('\tdone')


########################################################################################################################
# Reading, writing and dumping files
########################################################################################################################
def get_embeddings():
    """
    Load embedding matrix from file
    :return:
    """
    assert os.path.exists(FLAGS.embeddings_file), "File " + FLAGS.embeddings_file + " does not exist"
    print('Loading pre-trained embeddings from', FLAGS.embeddings_file)
    with open(FLAGS.embeddings_file, 'rb') as f:
        embedding_matrix = pickle.load(f)
    vocabulary_size, embedding_dimension = embedding_matrix.shape
    print('\n--- Loaded embeddings with vocabulary size    : {}\n'.format(vocabulary_size),
          '\t                  with embedding dimension: {}'.format(embedding_dimension),
          '\n\tfrom file:', FLAGS.embeddings_file)
    return embedding_matrix


########################################################################################################################
# Utils (Preprocess files)
########################################################################################################################
def inline_struct_types_in_file(data, dic):
    """
    Inline structure types in the whole file
    :param data: list of strings representing the content of one file
    :param dic: dictionary ["structure name", "corresponding literal structure"]
    :return: modified data
    """
    # Remove all "... = type {..." statements since we don't need them anymore
    data = [stmt for stmt in data if not re.match('.* = type ', stmt)]

    # Inline the named structures throughout the file
    for i in range(len(data)):

        possible_struct = re.findall('(' + rgx.struct_name + ')', data[i])
        if len(possible_struct) > 0:
            for s in possible_struct:
                if s in dic and not re.match(s + r'\d* = ', data[i]):
                    # Replace them by their value in dictionary
                    data[i] = re.sub(re.escape(s) + rgx.struct_lookahead, dic[s], data[i])

    return data


def inline_struct_types_txt(data, data_with_structure_def):
    """
    Inline structure types so that the code has no more named structures but only explicit aggregate types
    And construct a dictionary of these named structures
    :param data: input data as a list of files where each file is a list of strings
    :return: data: modified input data
             dictio: list of dictionaries corresponding to source files,
                     where each dictionary has entries ["structure name", "corresponding literal structure"]
    """
    print('\tConstructing dictionary of structures and inlining structures...')
    dictio = defaultdict(list)

    # Loop on all files in the dataset
    for i in range(len(data)):
        # Construct a dictionary ["structure name", "corresponding literal structure"]
        data_with_structure_def[i], dict_temp = \
            i2v_prep.construct_struct_types_dictionary_for_file(data_with_structure_def[i])

        # If the dictionary is empty
        if not dict_temp:
            found_type = False
            for l in data[i]:
                if re.match(rgx.struct_name + ' = type (<?\{ .* \}|opaque|{})', l):
                    found_type = True
                    break
            assert not found_type, "Structures' dictionary is empty for file containing type definitions: \n" + \
                                   data[i][0] + '\n' + data[i][1] + '\n' + data[i] + '\n'

        # Use the constructed dictionary to substitute named structures
        # by their corresponding literal structure throughout the program
        data[i] = inline_struct_types_in_file(data[i], dict_temp)

        # Add the entries of the dictionary to the big dictionary
        for k, v in dict_temp.items():
            dictio[k].append(v)

    return data, dictio


def abstract_statements_from_identifiers_txt(data):
    """
    Simplify lines of code by stripping them from their identifiers,
    unnamed values, etc. so that LLVM IR statements can be abstracted from them
    :param data: input data as a list of files where each file is a list of strings
    :return: modified input data
    """
    data = remove_local_identifiers(data)
    data = remove_global_identifiers(data)
    data = remove_labels(data)
    data = replace_unnamed_values(data)
    data = remove_index_types(data)

    return data


def remove_local_identifiers(data):
    """
    Replace all local identifiers (%## expressions) by "<%ID>"
    :param data: input data as a list of files where each file is a list of strings
    :return: modified input data
    """
    print('\tRemoving local identifiers ...')
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = re.sub(rgx.local_id, "<%ID>", data[i][j])

    return data


def remove_global_identifiers(data):
    """
    Replace all local identifiers (@## expressions) by "<@ID>"
    :param data: input data as a list of files where each file is a list of strings
    :return: modified input data
    """
    print('\tRemoving global identifiers ...')
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = re.sub(rgx.global_id, "<@ID>", data[i][j])

    return data


def remove_labels(data):
    """
    Replace label declarations by token '<LABEL>'
    :param data: input data as a list of files where each file is a list of strings
    :return: modified input data
    """
    print('\tRemoving labels ...')
    for i in range(len(data)):
        for j in range(len(data[i])):
            if re.match(r'; <label>:\d+:?(\s+; preds = )?', data[i][j]):
                data[i][j] = re.sub(r":\d+", ":<LABEL>", data[i][j])
                data[i][j] = re.sub("<%ID>", "<LABEL>", data[i][j])
            elif re.match(rgx.local_id_no_perc + r':(\s+; preds = )?', data[i][j]):
                data[i][j] = re.sub(rgx.local_id_no_perc + ':', "<LABEL>:", data[i][j])
                data[i][j] = re.sub("<%ID>", "<LABEL>", data[i][j])
            if '; preds = ' in data[i][j]:
                s = data[i][j].split('  ')
                if s[-1][0] == ' ':
                    data[i][j] = s[0] + s[-1]
                else:
                    data[i][j] = s[0] + ' ' + s[-1]

    return data


def replace_unnamed_values(data):
    """
    Replace unnamed_values by abstract token:
        integers: <INT>
        floating points: <FLOAT> (whether in decimal or hexadecimal notation)
        string: <STRING>
    :param data: input data as a list of files where each file is a list of strings
    :return: modified input data
    """
    print('\tRemoving immediate values ...')
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = re.sub(r' ' + rgx.immediate_value_float_hexa, " <FLOAT>", data[i][j])  # hexadecimal notation
            data[i][j] = re.sub(r' ' + rgx.immediate_value_float_sci, " <FLOAT>", data[i][j])  # decimal / scientific
            if re.match("<%ID> = extractelement", data[i][j]) is None and \
                re.match("<%ID> = extractvalue", data[i][j]) is None and \
                re.match("<%ID> = insertelement", data[i][j]) is None and \
                re.match("<%ID> = insertvalue", data[i][j]) is None:
                data[i][j] = re.sub(r'(?<!align)(?<!\[) ' + rgx.immediate_value_int, " <INT>", data[i][j])

            data[i][j] = re.sub(rgx.immediate_value_string, " <STRING>", data[i][j])

    return data


def remove_index_types(data):
    """
    Replace the index type in expressions containing "extractelement" or "insertelement" by token <TYP>
    :param data: input data as a list of files where each file is a list of strings
    :return: modified input data
    """
    print('\tRemoving index types ...')
    for i in range(len(data)):
        for j in range(len(data[i])):
            if re.match("<%ID> = extractelement", data[i][j]) is not None or \
                re.match("<%ID> = insertelement", data[i][j]) is not None:
                data[i][j] = re.sub(r'i\d+ ', '<TYP> ', data[i][j])

    return data


########################################################################################################################
# Transform a folder of raw IR into trainable data to be used as input data in tasks
########################################################################################################################
def llvm_ir_to_trainable(folder_ir):
    ####################################################################################################################
    # Setup
    assert len(folder_ir) > 0, "Please specify a folder containing the raw LLVM IR"
    assert os.path.exists(folder_ir), "Folder not found: " + folder_ir
    folder_seq = re.sub('ir', 'seq', folder_ir)
    if len(folder_seq) > 0:
        print('Preparing to write LLVM IR index sequences to', folder_seq)
        if not os.path.exists(folder_seq):
            os.makedirs(folder_seq)

    # Get sub-folders if there are any
    listing = os.listdir(folder_ir + '/')
    folders_ir = list()
    folders_seq = list()
    found_subfolder = False
    for path in listing:
        if os.path.isdir(os.path.join(folder_ir, path)):
            folders_ir.append(os.path.join(folder_ir, path))
            folders_seq.append(os.path.join(folder_seq, path))
            found_subfolder = True
    if found_subfolder:
        print('Found', len(folders_ir), 'subfolders')
    else:
        print('No subfolders found in', folder_ir)
        folders_ir = [folder_ir]
        folders_seq = [folder_seq]

    # Loop over sub-folders
    summary = ''
    num_folders = len(folders_ir)
    for i, raw_ir_folder in enumerate(folders_ir):

        l = folders_seq[i] + '/'
        if not os.path.exists(l) or len(os.listdir(l)) == 0:

            ############################################################################################################
            # Read files

            # Read data from folder
            print('\n--- Read data from folder ', raw_ir_folder)
            raw_data, file_names = i2v_prep.read_data_files_from_folder(raw_ir_folder)

            # Print data statistics and release memory
            source_data_list, source_data = i2v_prep.data_statistics(raw_data, descr="reading data from source files")
            del source_data_list

            # Source code transformation: simple pre-processing
            print('\n--- Pre-process code')
            preprocessed_data, functions_declared_in_files = i2v_prep.preprocess(raw_data)
            preprocessed_data_with_structure_def = raw_data

            ############################################################################################################
            # Load vocabulary and cut off statements

            # Vocabulary files
            folder_vocabulary = FLAGS.vocabulary_dir
            dictionary_pickle = os.path.join(folder_vocabulary, 'dic_pickle')
            cutoff_stmts_pickle = os.path.join(folder_vocabulary, 'cutoff_stmts_pickle')

            # Load dictionary and cutoff statements
            print('\tLoading dictionary from file', dictionary_pickle)
            with open(dictionary_pickle, 'rb') as f:
                dictionary = pickle.load(f)
            print('\tLoading cut off statements from file', cutoff_stmts_pickle)
            with open(cutoff_stmts_pickle, 'rb') as f:
                stmts_cut_off = pickle.load(f)
            stmts_cut_off = set(stmts_cut_off)

            ############################################################################################################
            # IR processing (inline structures, abstract statements)

            # Source code transformation: inline structure types
            print('\n--- Inline structure types')
            processed_data, structures_dictionary = inline_struct_types_txt(preprocessed_data,
                                                                            preprocessed_data_with_structure_def)

            # Source code transformation: identifier processing (abstract statements)
            print('\n--- Abstract statements from identifiers')
            processed_data = abstract_statements_from_identifiers_txt(processed_data)

            ############################################################################################################
            # Write indexed sequence of statements
            seq_folder = folders_seq[i]
            if not os.path.exists(seq_folder):
                os.makedirs(seq_folder)

            # Write indexed sequence of statements to file
            unknown_counter_folder = list()
            seq_length_folder = list()
            file_counter = 0
            for file in processed_data:

                stmt_indexed = list()  # Construct indexed sequence
                unknown_counter = 0  # Reset unknown counter
                for stmt in file:

                    # check whether this is a label, in which case we ignore it
                    if re.match(r'((?:<label>:)?(<LABEL>):|; <label>:<LABEL>)', stmt):
                        continue

                    # check whether this is an unknown
                    if stmt in stmts_cut_off:
                        stmt = rgx.unknown_token
                        unknown_counter += 1

                    # lookup and add to list
                    if stmt not in dictionary.keys():
                        print("NOT IN DICTIONARY:", stmt)
                        stmt = rgx.unknown_token
                        unknown_counter += 1

                    stmt_indexed.append(dictionary[stmt])

                # Write to csv
                file_name_csv = os.path.join(seq_folder, file_names[file_counter][:-3] + '_seq.csv')
                file_name_rec = os.path.join(seq_folder, file_names[file_counter][:-3] + '_seq.rec')
                with open(file_name_csv, 'w') as csv, open(file_name_rec, 'wb') as rec:
                    for ind in stmt_indexed:
                        csv.write(str(ind) + '\n')
                        rec.write(struct.pack('I', int(ind)))
                print('\tPrinted data pairs to file', file_name_csv)
                print('\tPrinted data pairs to file', file_name_rec)
                print('\t#UNKS', unknown_counter)

                # Increment counter
                unknown_counter_folder.append(unknown_counter)
                seq_length_folder.append(len(stmt_indexed))
                file_counter += 1

            # Print stats
            out = '\n\nFolder: ' + raw_ir_folder + '(' + str(i) + '/' + str(num_folders) + ')'
            out += '\n\nNumber of files processed: ' + str(len(seq_length_folder))
            out += '\n--- Sequence length stats:'
            out += '\nMin seq length    : {}'.format(min(seq_length_folder))
            out += '\nMax seq length    : {}'.format(max(seq_length_folder))
            out += '\nAvg seq length    : {}'.format(sum(seq_length_folder) / len(seq_length_folder))
            out += '\nTotal number stmts: {}'.format(sum(seq_length_folder))
            out += '\n--- UNK count stats:'
            out += '\nMin #UNKS in a sequence  : {}'.format(min(unknown_counter_folder))
            out += '\nMax #UNKS in a sequence  : {}'.format(max(unknown_counter_folder))
            out += '\nAvg #UNKS in a sequence  : {}'.format(sum(unknown_counter_folder) / len(unknown_counter_folder))
            out += '\nSum #UNKS in all sequence: {} / {}, {}%'.format(sum(unknown_counter_folder),
                                                                      sum(seq_length_folder),
                                                                      sum(unknown_counter_folder) * 100 / sum(
                                                                          seq_length_folder))
            print(out)
            summary += '\n' + out

    # When all is done, print a summary:
    print(summary)
    return folder_seq
