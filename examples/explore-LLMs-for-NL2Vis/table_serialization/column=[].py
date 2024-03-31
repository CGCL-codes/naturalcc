import os
import time

os.environ["OPENAI_API_KEY"] = ""
import openai
import tqdm
from gensim import corpora, models, similarities
import sqlite3
import re
import numpy as np
import math
import pandas as pd

from scipy import stats
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    # 转换为小写，分词
    words = word_tokenize(text.lower())
    # 移除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return words


def calculate_similarity(question_tokens, row_tokens):
    question_set = set(question_tokens)
    row_set = set(row_tokens)
    intersection = question_set.intersection(row_set)
    similarity = len(intersection) / (len(question_set) + len(row_set) - len(intersection))
    return similarity

def load_data(encoder_file, decoder_file):
    with open(encoder_file, 'r', encoding='gb18030') as f:
        encoder_questions = f.readlines()
    with open(decoder_file, 'r', encoding='gb18030') as f:
        decoder_answers = f.readlines()
    return  encoder_questions, decoder_answers

def save_data(generated_queries, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for query in generated_queries:
            f.write(query + '\n')
    print(f"saved to {file_path}")

def model_api(model, prompt):
    # print(f"new prompt:\n {prompt}")
    if model == 'gpt-4' or model == 'gpt-3.5-turbo':
        message = [
            {"role": "user", "content": prompt}
        ]
        try:
            # response = openai.Completion.create(
            response = openai.ChatCompletion.create(
                model=model,  # gpt-3.5-turbo, text-davinci-003, text-davinci-002
                # temperature=0.7,
                # max_tokens=4100,
                # top_p=1,
                # frequency_penalty=0,
                # presence_penalty=0,
                messages=message,
                # max_tokens=1000,
                # n=1,
                # stop=None,
                # temperature=0.5,
            )
            generated_answer = response.choices[0]['message']['content'].strip().lower()
        except Exception as e:
            time.sleep(25)
            return model_api(model,prompt)
    else:
        try:
            response = openai.Completion.create(
                engine=model,  # gpt-3.5-turbo, text-davinci-003, text-davinci-002
                temperature=0.7,
                # max_tokens=4100,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                prompt=prompt,
                max_tokens=50,
                # n=1,
                # stop=None,
                # temperature=0.5,
            )
            generated_answer = response.choices[0].text.strip().lower()
        except Exception as e:
            time.sleep(25)
            return model_api(model,prompt)
    # print(generated_answer)
    return generated_answer


def get_schema(db, table_names, model, question, limit, fk):
    table_sep = ""
    table_str = "Table {table} , Columns = [{columns}]; "
    column_sep = " , "
    value_str = "Values = [{columns}]; "
    column_str_with_values = "{column} ( {values} )"
    column_str_without_values = "{column}"
    value_sep = " , "
    tables = ''
    for table_name in table_names:
        database_file = os.path.join('../data/raw_data/database', db, db + ".sqlite")
        conn = sqlite3.connect(database_file)
        conn.text_factory = str
        cursor = conn.cursor()

        foreign_keys = []
        if fk :
            # Extract each table's DDL
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND LOWER(name)=?;", tuple([table_name]))
            ddl = cursor.fetchone()[0].lower()
            # foreign_key_pattern = r"FOREIGN KEY \(`(\w+)`\) REFERENCES `(\w+)`\(`(\w+)`\)"
            foreign_key_pattern = r"FOREIGN KEY \(`?([^`)]+)`?\) REFERENCES `?([^`)]+)`?\(`?([^`)]+)`?\)"
            # foreign_key_pattern = r'FOREIGN KEY\(`?([^`)]+)`?\)\s*REFERENCES\s*`?([^`)]+)`?\(`?([^`)]+)`?\)'
            matches = re.findall(foreign_key_pattern, ddl, re.IGNORECASE)
            # if not matches:
            #     print(ddl)

            for match in matches:
                foreign_keys.append(f"{table_name}.{match[0]} = {match[1]}.{match[2]}")

        cursor.execute(f"SELECT * FROM {table_name}")
        column_data = []
        column_names = [description[0].lower() for description in cursor.description]  # Extract column names
        for row in cursor.fetchall():
            column_values = [str(value).lower() for value in row]  # Extract column values
            column_data.append(column_values)

        question_tokens = preprocess(question)
        similarities = []
        for row in column_data:
            row_text = ' '.join(row)
            row_tokens = preprocess(row_text)
            similarity = calculate_similarity(question_tokens, row_tokens)
            similarities.append((similarity, row))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[0], reverse=True)
        table_data = []
        # Get the top 'limit' rows
        top_rows = similarities[:limit]
        for _, row in top_rows:
            # table_data.append(list(row))
            table_data.append([str(x).lower() for x in row])

        table = []
        table_sample = ''
        value_sample = ''
        for column_name in column_names:
            column_str = column_str_without_values.format(column=column_name)
            table.append(column_str)
        table_sample = table_str.format(table=table_name, columns=" , ".join(table))
        if table_data:
            table = []
            for column_name, column_values in zip(column_names, zip(*table_data)):
                values_str = ", ".join(column_values)
                column_str = column_str_with_values.format(column=column_name, values=values_str)
                table.append(column_str)
            value_sample = value_str.format(table=table_name, columns=" , ".join(table))
        foreign_keys = 'Foreign_keys = '+str(foreign_keys).replace('`', '') + '\n' if foreign_keys else ''
        tables = tables + table_sample + value_sample + str(foreign_keys)

    conn.close()
    return tables


def search_example_by_question(question, example_encoder_questions, example_decoder_answers,model,limit_table,fk,nshot):
    # 将和要查询的问题最相似的example提取出来作为prompt输入
    # 创建一个字典
    dictionary = corpora.Dictionary([text.split() for text in example_encoder_questions])
    # 创建语料库
    corpus = [dictionary.doc2bow(text.split()) for text in example_encoder_questions]
    # 训练TF-IDF模型
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # 将查询字符串转换为向量
    query_bow = dictionary.doc2bow(question.split())
    query_tfidf = tfidf[query_bow]
    # 计算与查询字符串的相似度
    index = similarities.MatrixSimilarity(corpus_tfidf)
    sims = index[query_tfidf]
    # 找到与查询字符串最相关的前1个字符串的下标
    top_indexes = sims.argsort()[-nshot:][::-1]
    example_data = ""
    tables = []
    for i in top_indexes:
        # 将问题和答案逐行组合成字符串
        query = example_encoder_questions[i].strip()
        g,db = example_decoder_answers[i].strip().split('\t')

        table_names = []
        from_tables = re.findall(r' FROM\s+([^\s,(]+)', g, re.IGNORECASE)
        join_tables = re.findall(r' JOIN\s+([^\s,]+)', g, re.IGNORECASE)
        table_names.extend(from_tables)
        table_names.extend(join_tables)
        table_names = [table_name.rstrip(')') for table_name in table_names]
        table_names = [table_name.lstrip('(') for table_name in table_names]
        table_names = list(set(table_names))
        db_str = ''
        for t in table_names:
            if t not in tables:
                tables.append(t)
                db_str = db_str + get_schema(db, [t], model, question, limit_table, fk)

        # db = get_schema(db,table_names,model,question,limit_table,fk)
        example_data += f"{db_str} \n"
        example_data += f"Question: {query}, \n"
        example_data += f"VQL: {g} \n"
    return example_data

def test(train_encoder_file,train_decoder_file,test_encoder_file,test_decoder_file,save_pre_file,save_target_file,model,sample_data,data_type,model_type,limit_table,fk,nshot):

    example_encoder_questions, example_decoder_answers = load_data(train_encoder_file, train_decoder_file)
    encoder_questions, decoder_answers = load_data(test_encoder_file, test_decoder_file)

    input_sample = encoder_questions
    target_sample = decoder_answers

    generated_answers = []
    count = 0
    pbar = tqdm.tqdm(input_sample)

    for i, question in enumerate(pbar):

        g,db = target_sample[i].strip().split('\t')
        # 把仅涉及到的表格提取出来

        table_names = []
        from_tables = re.findall(r' FROM\s+([^\s,(]+)', g, re.IGNORECASE)
        join_tables = re.findall(r' JOIN\s+([^\s,]+)', g, re.IGNORECASE)
        table_names.extend(from_tables)
        table_names.extend(join_tables)
        table_names = [table_name.rstrip(')') for table_name in table_names]
        table_names = [table_name.lstrip('(') for table_name in table_names]
        table_names = list(set(table_names))


        db = get_schema(db,table_names,model,question,limit_table,fk)

        example_data = search_example_by_question(question, example_encoder_questions, example_decoder_answers,model,limit_table,fk,nshot)
        question = '\nQuestion: ' + question + 'VQL:'
        prompt = example_data + db + question
        print(prompt)
        generated_answer = model_api(model,prompt)

        print(f"\nExpected answer: {g}")
        print(f"Generated answer: {generated_answer}")

        generated_answers.append(generated_answer)
        count += 1  # increment counter

        # save data every 300 iterations
        if count % 300 == 0:
            save_data(generated_answers, f'{model_type}_{data_type}_column=[]_data{limit_table}_fk{fk}_{nshot}shot_{count}.txt')

    # save any remaining data after loop finishes
    save_data(generated_answers, save_pre_file)

if __name__ == "__main__":
    # data_type = 'radn_split'
    data_type = 'final_processed'

    # model_type = 'gpt2'
    model_type = 'gpt3'
    # model_type = 'gpt3.5'
    # model_type = 'gpt4'
    model = "text-davinci-003"
    # model = 'gpt-4'

    limit_table = 0
    fk = 1
    nshot = 7
    train_encoder_file = f"../data/data_{data_type}/train/train_encode.txt"
    train_decoder_file = f"../data/data_{data_type}/train/train_decode_db.txt"
    valid_encoder_file = f"../data/data_{data_type}/dev/dev_encode.txt"
    valid_decoder_file = f"../data/data_{data_type}/dev/dev_decode_db.txt"
    test_encoder_file = f"../data/data_{data_type}/test/test_encode.txt"
    test_decoder_file = f"../data/data_{data_type}/test/test_decode_db.txt"
    save_pre_file = f'{model_type}_{data_type}_column=[]_data{limit_table}_fk{fk}_{nshot}shot_total.txt'
    print(save_pre_file)

    test(train_encoder_file,train_decoder_file,test_encoder_file,test_decoder_file,save_pre_file,model,data_type,model_type,limit_table,fk,nshot)