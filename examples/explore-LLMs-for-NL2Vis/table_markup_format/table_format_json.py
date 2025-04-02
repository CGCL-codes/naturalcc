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

from scipy import stats
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
VALUE_NUM_SYMBOL = "{value}"
def strip_query(query):
    '''
    return keywords of sql query
    '''
    query_keywords = []
    query = query.strip().replace(";", "").replace("\t", "")
    query = query.replace("(", " ( ").replace(")", " ) ")
    query = query.replace(">=", " >= ").replace("<=", " <= ").replace("!=", " != ").replace("=", " = ")

    # then replace all stuff enclosed by "" with a numerical value to get it marked as {VALUE}
    str_1 = re.findall("\"[^\"]*\"", query)
    str_2 = re.findall("\'[^\']*\'", query)

    values = str_1 + str_2
    for val in values:
        query = query.replace(val.strip(), VALUE_NUM_SYMBOL)

    query_tokenized = query.split()
    float_nums = re.findall("[-+]?\d*\.\d+", query)
    query_tokenized = [VALUE_NUM_SYMBOL if qt in float_nums else qt for qt in query_tokenized]
    query = " ".join(query_tokenized)
    int_nums = [i.strip() for i in re.findall("[^tT]\d+", query)]

    query_tokenized = [VALUE_NUM_SYMBOL if qt in int_nums else qt for qt in query_tokenized]
    # print int_nums, query, query_tokenized

    for tok in query_tokenized:
        if "." in tok:
            table = re.findall("[Tt]\d+\.", tok)
            if len(table) > 0:
                to = tok.replace(".", " . ").split()
                to = [t.lower() for t in to if len(t) > 0]
                query_keywords.extend(to)
            else:
                query_keywords.append(tok.lower())

        elif len(tok) > 0:
            query_keywords.append(tok.lower())
    query_keywords = [w for w in query_keywords if len(w) > 0]
    query_sentence = " ".join(query_keywords)
    query_sentence = query_sentence.replace("> =", ">=").replace("! =", "!=").replace("< =", "<=")
    #     if '>' in query_sentence or '=' in query_sentence:
    #        print query_sentence
    return query_sentence.split()

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
                max_tokens=1000,
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

def get_schema(db, table_names, model,question,limit):

    tables = []
    for table_name in table_names:
        database_file = os.path.join('../data/raw_data/database', db, db + ".sqlite")
        conn = sqlite3.connect(database_file)
        conn.text_factory = str
        cursor = conn.cursor()
        print(table_name)
        cursor.execute(f"SELECT * FROM {table_name}")
        column_names = [str(description[0]).lower() for description in cursor.description]
        rows = cursor.fetchall()

        question_tokens = preprocess(question)
        similarities = []
        for row in rows:
            row_text = ' '.join(str(x) for x in row)
            row_tokens = preprocess(row_text)
            similarity = calculate_similarity(question_tokens, row_tokens)
            similarities.append((similarity, row))

        # 按相似度降序排列并获取前limit行
        top_rows = sorted(similarities, key=lambda x: x[0], reverse=True)[:limit]

        table_data = [column_names]
        for _, row in top_rows:
            # table_data.append(list(row))
            table_data.append([str(x).lower() for x in row])

        data = {
            'TableName': table_name,
            "Column": column_names,
            'Data': table_data
        }
        tables.append(data)
    conn.close()
    return str(tables)

def search_example_by_question(question, example_encoder_questions, example_decoder_answers,model,limit_table):
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
    top_indexes = sims.argsort()[-1:][::-1]
    example_data = ""
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

        db = get_schema(db,table_names,model,query,limit_table)
        example_data += f"{db} \n"
        example_data += f"Question: {query}, \n"
        example_data += f"VQL: {g} \n"
    return example_data

def test(train_encoder_file,train_decoder_file,test_encoder_file,test_decoder_file,save_pre_file,model,data_type,model_type,limit_table):

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
        vql = " ".join(strip_query(g))
        table_names = []
        from_tables = re.findall(r' FROM\s+([^\s,(]+)', vql, re.IGNORECASE)
        join_tables = re.findall(r' JOIN\s+([^\s,]+)', vql, re.IGNORECASE)
        table_names.extend(from_tables)
        table_names.extend(join_tables)
        table_names = [table_name.rstrip(')') for table_name in table_names]
        table_names = [table_name.lstrip('(') for table_name in table_names]
        table_names = list(set(table_names))


        db = get_schema(db,table_names,model,question,limit_table)

        example_data = search_example_by_question(question, example_encoder_questions, example_decoder_answers,model,limit_table)
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
            save_data(generated_answers, f'{model_type}_{data_type}_json_data{limit_table}_oneshot_{count}.txt')

    # save any remaining data after loop finishes
    save_data(generated_answers, save_pre_file)

if __name__ == "__main__":
    data_type = 'radn_split'
    # data_type = 'final_processed'

    # model_type = 'gpt2'
    model_type = 'gpt3'
    # model_type = 'gpt3.5'
    # model_type = 'gpt4'
    model = "text-davinci-003"
    # model = "gpt-4"

    limit_table = 3

    train_encoder_file = f"../data/data_{data_type}/train/train_encode.txt"
    train_decoder_file = f"../data/data_{data_type}/train/train_decode_db.txt"
    valid_encoder_file = f"../data/data_{data_type}/dev/dev_encode.txt"
    valid_decoder_file = f"../data/data_{data_type}/dev/dev_decode_db.txt"
    test_encoder_file = f"../data/data_{data_type}/test/test_encode.txt"
    test_decoder_file = f"../data/data_{data_type}/test/test_decode_db.txt"
    save_pre_file = f'{model_type}_{data_type}_json_data{limit_table}_oneshot_total.txt'


    test(train_encoder_file,train_decoder_file,test_encoder_file,test_decoder_file,save_pre_file,model,data_type,model_type,limit_table)
