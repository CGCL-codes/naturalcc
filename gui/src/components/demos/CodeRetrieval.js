import React from 'react';
import styled from 'styled-components';
import { withRouter } from 'react-router-dom';
import { Collapse } from 'antd';

import HeatMap from '../HeatMap';
import Model from '../Model';
import OutputField from '../OutputField';
import CodeDisplay from '../CodeDisplay'
import SyntaxHighlight from '../highlight/SyntaxHighlight.js';

const title = 'Code Retrieval';

const description = (
    <span>
        <span>
            Searching  semantically  similar  codesnippets given a natural language query can provide developersa series of templates for reference for rapid prototyping.
            <p>Dataset: <a href="https://github.com/github/codesearchnet#data">CodeSearchNet(ruby)</a></p>
            {/* <p><i>*Code is running on 2 core cpu. If it is slow, please wait. Thanks!*</i></p> */}
        </span>
    </span>
);

const taskModels = [
    {
        name: "Conv1d",
        modelId:"Conv1d"
    },
    {
        name: "NBOW",
    //     desc: <span>
    //   This model is the baseline model described
    //   in <a href="https://arxiv.org/pdf/1409.3215.pdf">Sequence to Sequence Learning with Neural Networks</a>.
    //   It uses a RNN based encoder as well as a RNN based encoder for text generation task.
    //   </span>,
        modelId: "NBOW",
        // usage: buildUsage("fine-grained-ner.2020-06-24.tar.gz")
    },
    {
        name: "BiRNN",
    //     desc: <span>
    //   Transformer, proposed in <a href="https://arxiv.org/abs/1603.01360">Attention Is All You Need</a>,
    //          employs self-attention for neural machine translation task .
    //   </span>,
        modelId: "BiRNN",
        // usage: buildUsage("fine-grained-ner.2020-06-24.tar.gz")
    },
    {
        name: "SelfAttn",
//         desc: <span>
//             This model is the baseline model described
//       in <a href="https://ai.nju.edu.cn/_upload/tpl/04/10/1040/template1040/publications/ijcai17-clone.pdf">Supervised Deep Features for Software Functional
// Clone Detection by Exploiting Lexical and Syntactical Information in Source
// Code.</a>.
//         </span>,
        modelId: "SelfAttn"
    },
]

const fields = [
    {
        name: 'utterance',
        label: 'Query',
        type: 'TEXT_AREA',
        placeholder: "Copy assets to dmg",
    },
    {name: "model", label: "Model", type: "RADIO", options: taskModels, optional: true}
];

const Output = ({ responseData }) => {
    const {
        predicted_sql_query,
    } = responseData;

    let code_snippet, internals;

    // if (predicted_sql_query.length > 1) {
    //     query = <SyntaxHighlight>{predicted_sql_query}</SyntaxHighlight>;
    // } else {
    //     query = <p>No query found!</p>;
    //     internals = null;
    // }
    code_snippet = <SyntaxHighlight language='python'>{predicted_sql_query}</SyntaxHighlight>;
    internals = null;
    return (
        <div className="model__content answer">
            <OutputField label="Retrieved Code" suppressSummary>
                <CodeDisplay value={eval(predicted_sql_query)}/> # print code rather than string type
                {/* {code_snippet} */}
            </OutputField>
            {internals}
        </div>
    );
};

const PanelDesc = styled.div`
    margin-bottom: ${({ theme }) => theme.spacing.sm};
`;

const examples = [
    {
        order:1,
        utterance: "Copy assets to dmg",
    },
    {
        order:2,
        utterance: "Checks transitive dependency licensing errors for the given software",
    },
];

const apiUrl = () => `/api/retrieve`;

const modelProps = { apiUrl, title, description, fields, examples, Output };

export default withRouter((props) => <Model {...props} {...modelProps} />);
