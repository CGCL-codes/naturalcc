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
    return encoder_questions, decoder_answers


def save_data(generated_queries, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for query in generated_queries:
            f.write(query + '\n')
    print(f"saved to {file_path}")


def model_api(model, prompt, max=2000):
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
            return model_api(model, prompt,)

    elif model == 'code-davinci-edit-001':
        try:
            response = openai.Edit.create(
                model="code-davinci-edit-001",
                input=prompt,
                instruction="",
                temperature=0,
                top_p=1
            )
            generated_answer = response.choices[0]['message']['content'].strip().lower()
        except Exception as e:
            time.sleep(25)
            return model_api(model, prompt)
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
                max_tokens=max,
                # n=1,
                # stop=None,
                # temperature=0.5,
            )
            generated_answer = response.choices[0].text.strip().lower()
        except Exception as e:

            if e.error['type'] == 'invalid_request_error':
                print('error')
                max = max - 200
            time.sleep(15)
            return model_api(model, prompt, max)
    # print(generated_answer)
    return generated_answer


def get_schema(db, table_names, model, question, limit):
    # db to py
    tables = []
    ddl_create = []
    for table_name in table_names:
        database_file = os.path.join('../data/raw_data/database', db, db + ".sqlite")
        conn = sqlite3.connect(database_file)
        conn.text_factory = str
        cursor = conn.cursor()

        # 提取每个表的DDL
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND LOWER(name)=?;", tuple([table_name]))
        ddl = cursor.fetchone()[0]

        # 将DDL列表转换为字符串变量

        # 获取最相关的数据
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

        table_data = []
        for _, row in top_rows:
            # table_data.append(list(row))
            table_data.append([str(x).lower() for x in row])

        data = {
            'TableName': table_name,
            "Column": column_names,
            'Data': table_data,
        }

        tables.append(data)
        ddl_create.append(ddl)

    table = '\n DDL Create: ' + ''.join(ddl_create).lower() + '\n Data: ' + str(tables)

    # 描述面向对象编程问题
    pre_code = '''
    # A Constraint is a rule that specifies the limits or conditions on a column.
    class Constraint:
        pass

    # A Column is a set of data values of a particular type that are stored in a table. A column can have a name, and optionally a constraint. A table can have one or more columns. For example, if you have a table called Students with columns Student_ID, Student_Name, Age, and Gender, each column can store a different type of information about the student.
    class Column:
        def __init__(self, name, PK_attribute: Constraint = None, FK_attribute: Constraint = None):
            self.name = name
            self.attribute1 = PK_attribute
            self.attribute2 = FK_attribute

    # A Table is a collection of related data organized in rows and columns. A table has a name and can have one or more columns, each with a name. A table can store data about a specific topic, such as customers, products, orders. For example, if you have a table called Customers with columns CustomerID, CustomerName, ContactName, Address, and City, each row in the table can store information about one customer.

    # A Record is a single row of data in a table. A record consists of one or more fields, each with a value that corresponds to a column in the table.
    class Record:
        def __init__(self, content:List):
            self.content = content

    class Table:
        def __init__(self, name, column_list: List[Column], data_list: List[Record]):
            self.name = name
            self.column_list = column_list
            self.data_list = data_list

    # Primary_Key is a constraint that uniquely identifies each record in a table. A table can have only one primary key, and it can consist of single or multiple columns. A primary key helps you to compare, sort and store records, and to create relationships between records.
    class Primary_Key(Constraint):
        pass

    # Foreign_Key is a constraint that links two tables together by referencing the primary key of another table. A foreign key helps to create relationships between tables and to enforce referential integrity.
    class Foreign_Key(Constraint):
        def __init__(self, referenced_table:Table, referenced_column:"the name of the column with primary key constraint of the referenced_table"):
            super(Foreign_Key, self).__init__()
            self.referenced_table = referenced_table
            self.referenced_column = referenced_column

    ---
'''
    ex_code = '''
    people_column_list = [Column("people_id",Primary_Key()),Column("name"),Column("country"),Column("is_male"),Column("age")]

    people_data_list = [
        Record([
            1,
            "mike weir",
            "canada",
            "t",
            34
        ]),
        Record([
            2,
            "juli hanson",
            "sweden",
            "f",
            32
        ]),
        Record([
            3,
            "ricky barnes",
            "united states",
            "t",
            30
        ])
    ]

    people = Table("people",people_column_list,people_data_list)

    # this Foreign key means the field refers to the primary key "people_id" of table people
    FK_people_id = Foreign_Key(people,"people_id")

    # both Column "male_id" and "female_id" refer to "people_id".
    wedding_column_list = [Column("church_id",Primary_key()),Column("male_id",Primary_key(),FK_people_id),Column("female_id",Primary_key(),FK_people_id),Column("is_male"),Column("year")]

    wedding_data_list = [
    Record([
        1,
        1,
        2,
        2014
    ]),
    Record([
        3,
        3,
        4,
        2015
    ]),
    Record([
        5,
        5,
        6,
        2016
    ])
]

wedding = Table("wedding",wedding_column_list,wedding_data_list)
'''
    ex_data = [
                  {
                      "TableName": "people",
                      "Column": [
                          "people_id",
                          "name",
                          "country",
                          "is_male",
                          "age"
                      ],
                      "Data": [
                          [
                              1,
                              "mike weir",
                              "canada",
                              "t",
                              34
                          ],
                          [
                              2,
                              "juli hanson",
                              "sweden",
                              "f",
                              32
                          ],
                          [
                              3,
                              "ricky barnes",
                              "united states",
                              "t",
                              30
                          ]
                      ]
                  },
                  {
                      "TableName": "wedding",
                      "Column": [
                          "church_id",
                          "male_id",
                          "female_id",
                          "year"
                      ],
                      "Data": [
                          [
                              1,
                              1,
                              2,
                              2014
                          ],
                          [
                              3,
                              3,
                              4,
                              2015
                          ],
                          [
                              5,
                              5,
                              6,
                              2016
                          ]
                      ]
                  }
              ],
    ex_create = '' \
                'CREATE TABLE "people" (' \
                '"People_ID" int,' \
                '"Name" text,' \
                '"Country" text,' \
                '"Is_Male" text,' \
                '"Age" int,' \
                'PRIMARY KEY ("People_ID")' \
                ');' \
                'CREATE TABLE "wedding"(' \
                '"Church_ID" int,' \
                '"Male_ID" int,' \
                '"Female_ID" int,' \
                '"Year" int,' \
                'PRIMARY KEY("Church_ID", "Male_ID", "Female_ID"),' \
                'FOREIGN KEY("Church_ID") REFERENCES `church`("Church_ID"),' \
                'FOREIGN KEY("Male_ID") REFERENCES `people`("People_ID"),' \
                'FOREIGN KEY("Female_ID") REFERENCES `people`("People_ID")' \
                ');' \
                ''
    ex = '\n DDL Create: ' + ex_create + '\n Data: ' + str(ex_data) + '\n Code: ' + ex_code
    q = 'Describe this data based on python class above using object-oriented programming, just return instantiation. Do not change the table names and column names, just keep the underscores and capitalization. Return only the most concise and important code. This is an example:'
    prompt = pre_code + '\n' + q + '\n' + ex + 'Now, please describe this data in python using object-oriented programming, just return the instantiation: ' + table + '\n Code: '
    # " Describe this data in c++ using object-oriented programming, return only code,use at most 1500 words.Removing the include reference library code, eliminate the need to show the insert data and returns only the most concise and important code,don't have to return the main function"
    get_answer = model_api(model, prompt)
    conn.close()
    return 'Used data is described by python code:' + get_answer


def search_example_by_question(question, example_encoder_questions, example_decoder_answers, model, limit_table):
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
        g, db = example_decoder_answers[i].strip().split('\t')

        table_names = []
        from_tables = re.findall(r' FROM\s+([^\s,(]+)', g, re.IGNORECASE)
        join_tables = re.findall(r' JOIN\s+([^\s,]+)', g, re.IGNORECASE)
        table_names.extend(from_tables)
        table_names.extend(join_tables)
        table_names = [table_name.rstrip(')') for table_name in table_names]
        table_names = [table_name.lstrip('(') for table_name in table_names]
        table_names = list(set(table_names))

        db = get_schema(db, table_names, model, question, limit_table)

        example_data += f"{db} \n"
        example_data += f"Question: {query}, \n"
        example_data += f"VQL: {g} \n"
    return example_data


def test(train_encoder_file, train_decoder_file, test_encoder_file, test_decoder_file, save_pre_file,
         model, data_type, model_type, limit_table):
    example_encoder_questions, example_decoder_answers = load_data(train_encoder_file, train_decoder_file)
    encoder_questions, decoder_answers = load_data(test_encoder_file, test_decoder_file)

    input_sample = encoder_questions
    target_sample = decoder_answers
    generated_answers = []
    count = 0
    pbar = tqdm.tqdm(input_sample)

    for i, question in enumerate(pbar):

        g, db = target_sample[i].strip().split('\t')
        # 把仅涉及到的表格提取出来

        table_names = []
        from_tables = re.findall(r' FROM\s+([^\s,(]+)', g, re.IGNORECASE)
        join_tables = re.findall(r' JOIN\s+([^\s,]+)', g, re.IGNORECASE)
        table_names.extend(from_tables)
        table_names.extend(join_tables)
        table_names = [table_name.rstrip(')') for table_name in table_names]
        table_names = [table_name.lstrip('(') for table_name in table_names]
        table_names = list(set(table_names))

        db = get_schema(db, table_names, model, question, limit_table)
        pre_text = 'Please generate VQL based on python programming description of tabular data and question.'
        pre_code = '''
                    # A Constraint is a rule that specifies the limits or conditions on a column.
                    class Constraint:
                        pass

                    # A Column is a set of data values of a particular type that are stored in a table. A column can have a name, and optionally a constraint. A table can have one or more columns. For example, if you have a table called Students with columns Student_ID, Student_Name, Age, and Gender, each column can store a different type of information about the student.
                    class Column:
                        def __init__(self, name, PK_attribute: Constraint = None, PK_attribute: Constraint = None):
                            self.name = name
                            self.attribute1 = PK_attribute
                            self.attribute2 = FK_attribute

                    # A Table is a collection of related data organized in rows and columns. A table has a name and can have one or more columns, each with a name. A table can store data about a specific topic, such as customers, products, orders. For example, if you have a table called Customers with columns CustomerID, CustomerName, ContactName, Address, and City, each row in the table can store information about one customer.

                    # A Record is a single row of data in a table. A record consists of one or more fields, each with a value that corresponds to a column in the table.
                    class Record:
                        def __init__(self, content:List):
                            self.content = content

                    class Table:
                        def __init__(self, name, column_list: List[Column], data_list: List[Record]):
                            self.name = name
                            self.column_list = column_list
                            self.data_list = data_list

                    # Primary_Key is a constraint that uniquely identifies each record in a table. A table can have only one primary key, and it can consist of single or multiple columns. A primary key helps you to compare, sort and store records, and to create relationships between records.
                    class Primary_Key(Constraint):
                        pass

                    # Foreign_Key is a constraint that links two tables together by referencing the primary key of another table. A foreign key helps to create relationships between tables and to enforce referential integrity.
                    class Foreign_Key(Constraint):
                        def __init__(self, referenced_table:Table, referenced_column:"the name of the column with primary key constraint of the referenced_table"):
                            super(Foreign_Key, self).__init__()
                            self.referenced_table = referenced_table
                            self.referenced_column = referenced_column

                    ---
                '''
        example_data = search_example_by_question(question, example_encoder_questions, example_decoder_answers, model,
                                                  limit_table)
        question = '\nQuestion: ' + question + 'VQL:'
        prompt = pre_text + pre_code + 'This is an example: ' + example_data + 'Now, please generate VQL to answer this question based on this code description of table. Gnerate VQL in one line and begin with : visulaize ' + db + question
        print(prompt)
        generated_answer = model_api(model, prompt, 100)

        print(f"\nExpected answer: {g}")
        print(f"Generated answer: {generated_answer}")

        generated_answers.append(generated_answer)
        count += 1  # increment counter

        # save data every 300 iterations
        if count % 300 == 0:
            save_data(generated_answers,
                      f'{model_type}_{data_type}_python_data{limit_table}_oneshot_{count}.txt')

    # save any remaining data after loop finishes
    save_data(generated_answers, save_pre_file)



if __name__ == "__main__":
    data_type = 'radn_split'
    # data_type = 'final_processed'

    # model_type = 'gpt2'
    model_type = 'gpt3'
    # model_type = 'gpt3.5'
    # model_type = 'gpt4'
    # model_type = 'codex'
    model = "text-davinci-003"
    # model = 'gpt-4'
    # model = 'code-davinci-edit-001'

    limit_table = 3
    train_encoder_file = f"../data/data_{data_type}/train/train_encode.txt"
    train_decoder_file = f"../data/data_{data_type}/train/train_decode_db.txt"
    valid_encoder_file = f"../data/data_{data_type}/dev/dev_encode.txt"
    valid_decoder_file = f"../data/data_{data_type}/dev/dev_decode_db.txt"
    test_encoder_file = f"../data/data_{data_type}/test/test_encode.txt"
    test_decoder_file = f"../data/data_{data_type}/test/test_decode_db.txt"
    save_pre_file = f'./code/{model_type}_{data_type}_python_data{limit_table}_oneshot_total.txt'


    test(train_encoder_file, train_decoder_file, test_encoder_file, test_decoder_file, save_pre_file,
         model, data_type, model_type, limit_table)
