import React from 'react';
import styled from 'styled-components';
import {withRouter} from 'react-router-dom';
import {Collapse} from '@allenai/varnish';

import HeatMap from '../HeatMap'
import CodeDemo from '../CodeDisplay'
import Model from '../Model'
import OutputField from '../OutputField'
import SyntaxHighlight from '../highlight/SyntaxHighlight.js';
import ParaBlock from '../ParaBlock';

const title = "Code Summarization";

// description for task
const description = (
    <span>
        <span>
        Generating comments forcode snippets is an effective way for program understandingand facilitate the software development and maintenance.
            <p>Dataset: <a href="https://github.com/wasiahmad/NeuralCodeSum/blob/master/data/python/get_data.sh">Python-Doc</a></p>
            {/* <p><i>*Code is running on 2 core cpu. If it is slow, please wait. Thanks!*</i></p> */}
        </span>
    </span>
)

const probabilitiesNote = (
    <span>
    Note: 
    <br></br>
    1. You can print Tab or click the word in the left list to complete the code.
    <br></br>
    2. The prediction percentages are normalized across these five sequences. The true probabilities are lower.
    <br></br>
    3. Code is running on 2 core CPUs. If it is slow, please wait. Thanks!
    </span>
)

// TODO: NCC cli
// const bashCommand = (modelUrl) => {
//     return `echo '{"code": "def addition(a, b):\\n\\treturn a+b"}' | \\
// allennlp predict ${modelUrl} -`
// }

// TODO: NCC predictor
// const pythonCommand = (modelUrl) => {
//     return `from allennlp.predictors.predictor import Predictor
// import allennlp_models.rc
// predictor = Predictor.from_path("${modelUrl}")
// predictor.predict(
//   passage="The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano.",
//   question="Who stars in The Matrix?"
// )`
// }

// tasks that have only 1 model, and models that do not define usage will use this as a default
// undefined is also fine, but no usage will be displayed for this task/model
const defaultUsage = undefined;

// TODO: define model
// const buildUsage = (modelUrl, configPath) => {
//     // model file, *.pt
//     const fullModelUrl = `https://storage.googleapis.com/allennlp-public-models/${modelUrl}`;
//     // model config, *.yml
//     const fullConfigPath = `https://raw.githubusercontent.com/allenai/allennlp-models/v1.0.0/training_config/rc/${configPath}`;
//     return {
//         installCommand: 'pip install ncc==0.1.0',
//         bashCommand: bashCommand(fullModelUrl),
//         pythonCommand: pythonCommand(fullModelUrl),
//         evaluationCommand: `allennlp evaluate \\
//     ${fullModelUrl} \\
//     https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json`,
//         trainingCommand: `allennlp train ${fullConfigPath} -s output_path`
//     }
// }

// models
// Add your models here
const taskModels = [
    {
        name: "Transformer",
        desc: <span>
      Transformer, proposed in <a href="https://arxiv.org/abs/1603.01360">Attention Is All You Need</a>,
             employs self-attention for neural machine translation task .
      </span>,
        modelId: "Transformer",
        // usage: buildUsage("fine-grained-ner.2020-06-24.tar.gz")
    },
    {
        name: "Seq2Seq",
        desc: <span>
      This model is the baseline model described
      in <a href="https://arxiv.org/pdf/1409.3215.pdf">Sequence to Sequence Learning with Neural Networks</a>.
      It uses a RNN based encoder as well as a RNN based encoder for text generation task.
      </span>,
        modelId: "Seq2seq",
        // usage: buildUsage("fine-grained-ner.2020-06-24.tar.gz")
    },
//     {
//         name: "Tree2Seq",
//         desc: <span>
//             This model is the baseline model described
//       in <a href="https://ai.nju.edu.cn/_upload/tpl/04/10/1040/template1040/publications/ijcai17-clone.pdf">Supervised Deep Features for Software Functional
// Clone Detection by Exploiting Lexical and Syntactical Information in Source
// Code.</a>.
//         </span>,
//         modelId: "Tree2Seq"
//     },
]

const fields = [
    {
        name: "code", label: "Code", type: "CODE_DISPLAY",
        placeholder: `def _organize_states_for_post_update(base_mapper, states, uowtransaction):\n\treturn list(_connections_for_states(base_mapper, uowtransaction, states))`
    },
    {
        name:"groundTruth",label:"Ground Truth", type:"TEXT_AREA",
        placeholder: 'make an initial pass across a set of states for update corresponding to post_update .'
    },
    {name: "model", label: "Model", type: "RADIO", options: taskModels, optional: true}
]


const Output = ({responseData}) => {
    const {
        predicted_summary,
    } = responseData;

    let code_summary, internals;

    // if (predicted_sql_query.length > 1) {
    //     query = <SyntaxHighlight>{predicted_sql_query}</SyntaxHighlight>;
    // } else {
    //     query = <p>No query found!</p>;
    //     internals = null;
    // }
    code_summary = <SyntaxHighlight language='python'>{predicted_summary}</SyntaxHighlight>;
    internals = null;
    return (
        <div className="model__content answer">
            <OutputField label="Generated Code Summary" suppressSummary>
                {/* {code_summary} */}
                <ParaBlock value={predicted_summary} />
            </OutputField>
            {internals}
        </div>
    );
};

const PanelDesc = styled.div`
  margin-bottom: ${({theme}) => theme.spacing.sm};
`;

const examples = [
    {
        order:1,
        code: "def _organize_states_for_post_update(base_mapper, states, uowtransaction):\n\treturn list(_connections_for_states(base_mapper, uowtransaction, states))\n",
        groundTruth:"make an initial pass across a set of states for update corresponding to post_update ."
    },
    {
        order:2,
        code: `def test_outdated_editables_columns_flag(script, data):\n\tscript.pip('install', '-f', data.find_links, '--no-index', 'simple==1.0')\n\tresult = script.pip('install', '-e', 'git+https:\/\/github.com\/pypa\/pip-test-package.git@0.1#egg=pip-test-package')\n\tresult = script.pip('list', '-f', data.find_links, '--no-index', '--editable', '--outdated', '--format=columns')\n\tassert ('Package' in result.stdout)\n\tassert ('Version' in result.stdout)\n\tassert ('Location' in result.stdout)\n\tassert (os.path.join('src', 'pip-test-package') in result.stdout), str(result)`,
        groundTruth:"test the behavior of --editable --outdated flag in the list command ."
    },
    {
        order:3,
        code:`def translate_pattern(pattern, anchor=1, prefix=None, is_regex=0):
        if is_regex:
            if isinstance(pattern, str):
                return re.compile(pattern)
            else:
                return pattern
        if pattern:
            pattern_re = glob_to_re(pattern)
        else:
            pattern_re = ''
        if (prefix is not None):
            empty_pattern = glob_to_re('')
            prefix_re = glob_to_re(prefix)[:(- len(empty_pattern))]
            sep = os.sep
            if (os.sep == '\\'):
                sep = '\\\\'
            pattern_re = ('^' + sep.join((prefix_re, ('.*' + pattern_re))))
        elif anchor:
            pattern_re = ('^' + pattern_re)
        return re.compile(pattern_re)`,
        groundTruth:"translate a shell-like wildcard pattern to a compiled regular expression ."
    },
    {
        order:4,
        code:`def test_sobel_v_horizontal():
        (i, j) = np.mgrid[(-5):6, (-5):6]
        image = (i >= 0).astype(float)
        result = filters.sobel_v(image)
        assert_allclose(result, 0)`,
        groundTruth:"vertical sobel on a horizontal edge should be zero ."
    },
    {
        order:5,
        code:`def prewitt_h(image, mask=None):
        assert_nD(image, 2)
        image = img_as_float(image)
        result = convolve(image, HPREWITT_WEIGHTS)
        return _mask_filter_result(result, mask)`,
        groundTruth:"find the horizontal edges of an image using the prewitt transform ."
    }
];

const apiUrl = () => `/api/summarize`

const modelProps = {apiUrl, title, description, fields, examples, Output}

export default withRouter(props => <Model {...props} {...modelProps}/>)
