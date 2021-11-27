import React from 'react';
import styled from 'styled-components';

// Component representing an element of the sequence that has been fixed,
// with an X to "unchoose" it.
const Chosen = ({action, unchoose, idx}) => (
    <ChosenLI key={`${idx}-${action}`}>
        <Action>{action}</Action>
        { /* eslint-disable-next-line */ }
        <Unchooser role="img" aria-label="x" onClick={unchoose}>❌</Unchooser>
    </ChosenLI>
)

// Component representing an element of the sequence that can be selected,
// with a dropdown containing all the possible choices.
const ChoiceDropdown = ({predictedAction, choices, choose, idx}) => {

    const options = choices.map(([probability, action], i) => (
        <ChoiceLI key={`${idx}-${action}`} onClick={() => choose(action)}>
            <Choice>{probability.toFixed(3)} {action}</Choice>
        </ChoiceLI>
    ))

    return (
        <ChoicesLI className="choice-dropdown" key={idx}>
            <Action>{predictedAction}</Action>
            <ChoicesUL>
                {options}
            </ChoicesUL>
        </ChoicesLI>
    )
}


class BeamSearch extends React.Component {
    constructor(props) {
        super(props)

        const { inputState } = props
        const initialSequence = (inputState && inputState.initial_sequence) || []
        this.state = { initialSequence }
    }


    render() {
        const { initialSequence } = this.state
        const { bestActionSequence, choices, runSequenceModel } = this.props

        // To "unchoose" the choice at a given index, we just rerun the model
        // with initial_sequence truncaated at that point.
        const unchoose = (idx) => () => {
            runSequenceModel({initial_sequence: initialSequence.slice(0, idx)})
        }

        // To choose an action at an index, we start with the existing forced choices,
        // then fill in with the elements of the bestActionSequence, and finally add
        // the chosen action.
        const choose = (idx) => (action) => {
            const sequence = initialSequence.slice(0, idx)
            while (sequence.length < idx) {
                sequence.push(bestActionSequence[sequence.length])
            }

            sequence.push(action)
            runSequenceModel({initial_sequence: sequence})
        }

        // We only want to render anything if ``choices`` is defined;
        // that is, if we have a beam search result and if it's the new backend.
        if (choices) {
            const listItems = bestActionSequence.map((action, idx) => {
                // Anything in ``initialSequence`` has already been chosen.
                if (idx < initialSequence.length) {
                    return <Chosen key={idx} action={action} unchoose={unchoose(idx)}/>
                } else {
                // Otherwise we need to offer a choice dropdown, and we should sort
                // from highest probability to lowest probability.
                const timestepChoices = choices[idx]
                timestepChoices.sort((a, b) => (b[0] - a[0]))

                return <ChoiceDropdown key={idx}
                                       predictedAction={action}
                                       choices={timestepChoices}
                                       choose={choose(idx)}/>
                }
            })

            return (
                <div>
                    <label>
                        Interactive Beam Search
                    </label>
                    <BeamSearchUL>
                       {listItems}
                    </BeamSearchUL>
                </div>
            )
        } else {
            return null
        }
    }
}


const BeamSearchUL = styled.ul`
    font-size: 0.5em;
`

const BeamSearchLI = styled.li`
    display: inline-block;
    font-size: 1em;
    position: relative;
    display: inline-block;
    border: ${({theme}) => `1px solid ${theme.palette.border.main}`};
`


const ChosenLI = styled(BeamSearchLI)`
    background-color: lightgray;
    color: white;

    :hover a {
        color: black;
    }
`


const ChoicesLI = styled(BeamSearchLI)`
    :hover {
        background-color: blue;
    }

    :hover .predicted-action {
        color: white;
    }

    :hover ul {
        display: table;
        background-color: #f9f9f9;
        font-size: 1.0em;
        border: ${({theme}) => `1px solid ${theme.palette.common.black}`};
    }
`

const Unchooser = styled.span`
    cursor: pointer;
`

const Action = styled.a`
    cursor: default;
    color: #232323;
`

const ChoicesUL = styled.ul`
    display: none;
    position: absolute;
    z-index: 1;
    list-style-type: none;
    padding: ${({theme}) => theme.spacing.xxs};
    margin: ${({theme}) => theme.spacing.xxs};
    width: auto;
    clear: both;
`

const ChoiceLI = styled.li`
`

const Choice = styled.a`
    padding: ${({theme}) => theme.spacing.xxs};
`

export default BeamSearch
