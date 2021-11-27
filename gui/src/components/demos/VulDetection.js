import React from 'react'
import { withRouter } from 'react-router-dom';
import styled from 'styled-components';
import _ from 'lodash';
import { Collapse } from '@allenai/varnish';

import OutputField from '../OutputField'
import SaliencyMaps from '../Saliency'
import HotflipComponent, { HotflipPanel } from '../Hotflip'
import { FormField, FormLabel, FormTextArea } from '../Form';

// const apiUrl = () => `http://149.28.205.231:5002/api/code_prediction/predict`

const NAME_OF_INPUT_TO_ATTACK = "tokens"
const NAME_OF_GRAD_INPUT = "grad_input_1"
const title = "Vulnerability Detection";

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

const DEFAULT = "Test";


function trimRight(str) {
  return str.replace(/ +$/, '');
}

const DEFAULT_MODEL = "345M"

const description = (
  <span>
    <p>Under construction...</p>
  </span>
)

class App extends React.Component {
  constructor(props) {
    super(props)

    this.currentRequestId = 0;

    this.state = {
      output: DEFAULT,
      top_tokens: null,
      logits: null,
      probabilities: null,
      loading: false,
      error: false,
      model: DEFAULT_MODEL,
      interpretData: null,
      attackData: null
    }

    // this.choose = this.choose.bind(this)
    // this.debouncedChoose = _.debounce(this.choose, 1000)
    // this.setOutput = this.setOutput.bind(this)
    // this.runOnEnter = this.runOnEnter.bind(this)
    // this.interpretModel = this.interpretModel.bind(this)
    // this.attackModel = this.attackModel.bind(this)
  }

  render() {

    let requestData = {"sentence": this.state.output};
    // let interpretData = this.state.interpretData;
    // let attackData = this.state.attackData;
    let tokens = [];
    if (this.state.tokens === undefined) {
        tokens = [];
    }
    else {
        if (Array.isArray(this.state.tokens[0])) {
            tokens = this.state.tokens[0];
        }
        else {
            tokens = this.state.tokens;
        }
    }
    return (
      <Wrapper classname="model">
        <ModelArea className="model__content answer">
          <h2><span>{title}</span></h2>
          <span>{description}</span>
        </ModelArea>
      </Wrapper>
    )
  }
}



const modelProps = {}

export default withRouter(props => <App {...props} {...modelProps}/>)
