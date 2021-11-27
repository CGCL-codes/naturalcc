import React from 'react'
import {withRouter} from 'react-router-dom';
import styled from 'styled-components';
import _ from 'lodash';
import {Collapse} from '@allenai/varnish';
import {Tabs, Select, Typography} from '@allenai/varnish';
import {message} from 'antd';

import OutputField from '../OutputField'
import SaliencyMaps from '../Saliency'
import HotflipComponent, {HotflipPanel} from '../Hotflip'
import {FormField, FormLabel, FormTextArea, FormSelect} from '../Form';

// const apiUrl = () => `http://149.28.205.231:5002/api/code_prediction/predict`

const NAME_OF_INPUT_TO_ATTACK = "tokens"
const NAME_OF_GRAD_INPUT = "grad_input_1"
const title = "Program Language Modeling";

const Wrapper = styled.div`
  color: #232323;
  flex-grow: 1;
  font-size: 1em;
  background: ${({theme}) => theme.palette.background.light};
  overflow: scroll;

  @media(max-width: 500px) {
    margin: 0;
  }
`

const ModelArea = styled.div`
  background: ${({theme}) => theme.palette.common.white};
`

const Loading = styled.div`
  position: absolute;
  bottom: 1rem;
  right: 1rem;
  display: flex;
  align-items: center;
  font-size: 0.8em;
  color: #8c9296;
`

const Error = styled(Loading)`
  color: red;
`

const LoadingText = styled.div`
  padding-left: ${({theme}) => theme.spacing.xs};
`

const InputOutput = styled.div`
  display: flex;
  margin-top: ${({theme}) => theme.spacing.sm};

  @media(max-width: 500px) {
    display: block;
  }
`

const InputOutputColumn = styled(FormField)`
  flex: 1 1 50%;

  :first-child {
    padding-right: ${({theme}) => theme.spacing.md};
  }

  :last-child {
    padding-left: ${({theme}) => theme.spacing.md};
  }

  @media(max-width: 500px) {
    :first-child,
    :last-child {
      padding: 0;
    }

    :first-child {
      padding: ${({theme}) => `0 0 ${theme.spacing.md}`};
    }
  }
`

const TextInput = styled(FormTextArea)`
  display: block;
  width: 100%;
  font-size: 1.25em;
  min-height: 100px;
  border: 1px solid rgba(0, 0, 0, 0.2);
  padding: ${({theme}) => theme.spacing.md};
`

const ListItem = styled.li`
  margin: ${({theme}) => `0 0 ${theme.spacing.xs}`};
`

const ChoiceList = styled.ul`
  padding: 0;
  margin: 0;
  flex-wrap: wrap;
  list-style-type: none;
`

const ChoiceItem = styled.button`
  color: #2085bc;
  cursor: pointer;
  background: transparent;
  display: inline-flex;
  align-items: center;
  line-height: 1;
  font-size: 1.15em;
  border: none;
  border-bottom: ${({theme}) => `2px solid ${theme.palette.common.transparent}`};
`

const UndoButton = styled(ChoiceItem)`
  color: #8c9296;
  margin-bottom: ${({theme}) => theme.spacing.xl};
`

const Probability = styled.span`
  color: #8c9296;
  margin-right: ${({theme}) => theme.spacing.xs};
  font-size: 0.8em;
  min-width: 4em;
  text-align: right;
`

const Token = styled.span`
  font-weight: 600;
`

const taskModels = [
    {
        name: "GPT2",
      //   desc: <span>
      // GPT2, proposed in <a href="https://arxiv.org/abs/1603.01360">Attention Is All You Need</a>,
      //        employs self-attention for neural machine translation task .
      // </span>,
        desc: <span>
      </span>,
        modelId: "GPT2",
        // usage: buildUsage("fine-grained-ner.2020-06-24.tar.gz")
    },
    {
        name: "SeqRNN",
      //   desc: <span>
      // This model is the baseline model described
      // in <a href="https://arxiv.org/pdf/1409.3215.pdf">Sequence to Sequence Learning with Neural Networks</a>.
      // It uses a RNN based encoder as well as a RNN based encoder for text generation task.
      // </span>,
        desc: <span>
      </span>,
        modelId: "SeqRNN",
        // usage: buildUsage("fine-grained-ner.2020-06-24.tar.gz")
    },
]

const OptDesc = styled.div`
  max-width: ${({theme}) => theme.breakpoints.md};
  white-space: break-spaces;
`;

// const DEFAULT = "body_content = self._serialize.body(parameters, 'ServicePrincipalCreateParameters')\nrequest = self._client.post(url, query_parameters)\nresponse = self._client.send( ";
const DEFAULT = "@ register . filter\ndef lookup ( h , key ) :\n\ttry : return h [ key ]\n\t";
// target: # except KeyError: return ''
// const DEFAULT = "def upgrade ( ) :\n\top . add_column ( 'column' , sa . Column ( 'nb_max_cards' , sa . Integer ) )\ndef downgrade ( ) :\n\top .";
// target: # drop_column('column', 'nb_max_cards')

function addToUrl(output, choice) {
    if ('history' in window) {
        window.history.pushState(null, null, '?text=' + encodeURIComponent(output + (choice || '')))
    }
}

function loadFromUrl() {
    const params =
        document.location.search.substr(1).split('&').map(p => p.split('='));
    const text = params.find(p => p[0] === 'text');
    return Array.isArray(text) && text.length === 2 ? decodeURIComponent(text.pop()) : null;
}

function trimRight(str) {
    return str.replace(/ +$/, '');
}

const DEFAULT_MODEL = "345M"

const description = (
    <span>
    <p>Code completion, which predicts the next code element based on the previously written code, has become an essential tool in many IDEs. It can boost developers’programming productivity. </p>
        <p>Dataset: <a href="https://www.sri.inf.ethz.ch/py150">Py150</a></p>
        {/* <p><i>*Code is running on 2 core cpu. If it is slow, please wait. Thanks!*</i></p> */}
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


class App extends React.Component {
    constructor(props) {
        super(props)

        this.currentRequestId = 0;

        this.state = {
            output: loadFromUrl() || DEFAULT,
            top_tokens: null,
            logits: null,
            probabilities: null,
            loading: false,
            error: false,
            model: DEFAULT_MODEL,
            interpretData: null,
            attackData: null,
            selectedSubModel: "GPT2"
        }

        this.choose = this.choose.bind(this)
        this.debouncedChoose = _.debounce(this.choose, 1000)
        this.setOutput = this.setOutput.bind(this)
        // this.runOnEnter = this.runOnEnter.bind(this)
        // this.interpretModel = this.interpretModel.bind(this)
        // this.attackModel = this.attackModel.bind(this)
    }

    setOutput(evt) {
        const value = evt.target.value
        if (value) { // TODO(michaels): I shouldn't need to do this
            const trimmed = trimRight(value);

            const loading = trimmed.length > 0;

            this.setState({
                output: value,
                top_tokens: null,
                logits: null,
                probabilities: null,
                interpretData: null,
                attackData: null,
                loading: loading
            })

            this.debouncedChoose()
        } else { // Update text input without request to backend server
            this.setState({
                output: value,
                top_tokens: null,
                logits: null,
                probabilities: null,
                interpretData: null,
                attackData: null,
                loading: false
            })
        }
    }

    createRequestId() {
        const nextReqId = this.currentRequestId + 1;
        this.currentRequestId = nextReqId;
        return nextReqId;
    }

    handleSubModelChange = (val) => {
        this.setState({selectedSubModel: val});
    }

    componentDidMount() {
        this.choose()
        if ('history' in window) {
            window.addEventListener('popstate', () => {
                const fullText = loadFromUrl();
                const doNotChangeUrl = fullText ? true : false;
                const output = fullText || DEFAULT;
                this.setState({
                    output,
                    loading: true,
                    top_tokens: null,
                    logits: null,
                    probabilities: null,
                    model: this.state.model
                }, () => this.choose(undefined, doNotChangeUrl));
            })
        }
    }

    // Handler that indicates the next word if 'Tab' is pressed.
    runOnTab = e => {
        if (e.key === 'Tab') {
            e.preventDefault();
            e.stopPropagation();
            // Here runs the indication function
            if (this.state.top_tokens) {
                this.choose(this.state.top_tokens[0]);
            } else {
                message.error("Sorry, it can't predict now.")
            }
        }
    }

    choose(choice = undefined, doNotChangeUrl) {
        // strip trailing spaces
        const textAreaText = this.state.output;
        if (trimRight(textAreaText).length === 0) {
            this.setState({loading: false});
            return;
        }

        this.setState({loading: true, error: false})
        // TODO(mattg): this doesn't actually send the newline token to the model in the right way.
        // I'm not sure how to fix that.
        const cleanedChoice = choice === undefined ? undefined : choice.replace(/↵/g, '\n');

        const sentence = choice === undefined ? textAreaText : textAreaText + cleanedChoice
        const payload = {
            sentence: sentence,
            model: this.state.selectedSubModel
        }

        const currentReqId = this.createRequestId();

        if ('history' in window && !doNotChangeUrl) {
            addToUrl(this.state.output, cleanedChoice);
        }

        fetch("/api/predict", {
            method: "POST",
            headers: {
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        })
            .then(response => response.json())
            .then(data => {
                if (this.currentRequestId === currentReqId) {
                    // If the user entered text by typing don't overwrite it, as that feels
                    // weird. If they clicked it overwrite it
                    const output = choice === undefined ? this.state.output : data.output
                    this.setState({...data, output: sentence, loading: false})
                    this.requestData = output;
                }
            })
            .catch(err => {
                console.error('Error trying to communicate with the API:', err);
                this.setState({error: true, loading: false});
            });
    }

    // Temporarily (?) disabled
    // runOnEnter(e) {
    //   if (e.key === 'Enter') {
    //       e.preventDefault()
    //       e.stopPropagation()
    //       this.choose()
    //   }
    // }

    render() {

        let requestData = {"sentence": this.state.output};
        // let interpretData = this.state.interpretData;
        // let attackData = this.state.attackData;
        let tokens = [];
        if (this.state.tokens === undefined) {
            tokens = [];
        } else {
            if (Array.isArray(this.state.tokens[0])) {
                tokens = this.state.tokens[0];
            } else {
                tokens = this.state.tokens;
            }
        }
        return (
            <Wrapper classname="model">
                <ModelArea className="model__content answer">
                    <h2><span>{title}</span></h2>
                    <span>{description}</span>
                    <FormLabel>Model</FormLabel>
                    <FormSelect
                        style={{width: 390}}
                        value={this.state.selectedSubModel || taskModels[0].modelId}
                        onChange={this.handleSubModelChange}
                        dropdownMatchSelectWidth={false}
                        optionLabelProp="label"
                        listHeight={370}
                    >
                        {
                            taskModels.map((value, i) => (
                                <Select.Option key={value.modelId} value={value.modelId} label={value.name}>
                                    <>
                                        <Typography.BodyBold>{value.name}</Typography.BodyBold>
                                        <OptDesc>{value.desc}</OptDesc>
                                    </>
                                </Select.Option>
                            ))
                        }
                    </FormSelect>
                    <InputOutput>
                        <InputOutputColumn>
                            <FormLabel>Sentence:</FormLabel>
                            <TextInput type="text"
                                       autoSize={{minRows: 5, maxRows: 10}}
                                       value={this.state.output}
                                       onKeyDown={this.runOnTab}
                                       onChange={this.setOutput}/>
                            {this.state.loading ? (
                                <Loading>
                                    <img src="/assets/loading-bars.svg" width="25" height="25" alt="loading"/>
                                    <LoadingText>Loading</LoadingText>
                                </Loading>
                            ) : null}
                            {this.state.error ? (
                                <Error>
                                    <span role="img" aria-label="warning">️⚠</span> Something went wrong. Please try
                                    again.
                                </Error>
                            ) : null}
                        </InputOutputColumn>
                        <InputOutputColumn>
                            <FormLabel>Predictions:</FormLabel>
                            <Choices output={this.state.output}
                                     index={0}
                                     choose={this.choose}
                                     runOnTab={this.runOnTab}
                                     logits={this.state.logits}
                                     top_tokens={this.state.top_tokens}
                                     probabilities={this.state.probabilities}
                                     hidden={this.state.loading}/>
                        </InputOutputColumn>
                    </InputOutput>
                    <p>{probabilitiesNote}</p>
                </ModelArea>
            </Wrapper>
        )
    }
}


const formatProbability = (probs, idx) => {
    // normalize the displayed probabilities
    var sum = probs.reduce(function (a, b) {
        return a + b;
    }, 0);
    var prob = probs[idx] / sum
    prob = prob * 100
    return `${prob.toFixed(1)}%`
}

const Choices = ({output, index, logits, top_tokens, choose, probabilities, runOnTab}) => {
    if (!top_tokens) {
        return null
    }
    if (top_tokens.length <= index) {
        return null
    }
    if (probabilities.length <= index) {
        return null
    }

    const lis = top_tokens.map((word, idx) => {
        const prob = formatProbability(probabilities, idx)
        // get rid of CRs
        // const cleanWord = word.join('').replace(' ,', ',').replace(/\n/g, "↵")
        //     .replace(/Ġ/g, " ").replace(/Ċ/g, "↵")
        //
        // const displaySeq = cleanWord.slice(-1) == "." ? cleanWord : cleanWord.concat(' ...')
        const displaySeq = word
        return (
            <ListItem key={`${idx}-${word}`}>
                <ChoiceItem
                    onClick={() => choose(word)}>
                    <Probability>{prob}</Probability>
                    {' '}
                    <Token>{displaySeq}</Token>
                </ChoiceItem>
            </ListItem>
        )
    })

    const goBack = () => {
        window.history.back();
    }

    const goBackItem = (
        <ListItem key="go-back">
            {'history' in window ? (
                <UndoButton onClick={goBack}>
                    <Probability>←</Probability>
                    {' '}
                    <Token>Undo</Token>
                </UndoButton>
            ) : null}
        </ListItem>
    )

    return (
        <ChoiceList>
            {lis}
            {goBackItem}
        </ChoiceList>
    )
}

const modelProps = {
    options: taskModels,
}

export default withRouter(props => <App {...props} {...modelProps}/>)
