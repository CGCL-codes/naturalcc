import React from 'react';
import styled from 'styled-components';
import { Button, Select, Radio } from '@allenai/varnish'
import RightOutlined from '@ant-design/icons/RightOutlined';

import BeamSearch from './BeamSearch'
import ParaBlock from './ParaBlock'
import { ImageParamControl, blobToString } from './ImageParamControl';
import { Tooltip } from './Shared'
import '../css/Button.css'
import { FormField, FormLabel, FormInput, FormTextArea, FormSelect } from './Form';
import CodeDemo from './CodeDisplay'

const PATTERN_NON_WORD_CHAR = /\W/;
const PATTERN_WORD_CHAR = /\w/;
const ELLIPSIS = '…';
const EXAMPLE_NAME_SEPARATOR = '@@';
const DEFAULT_OPTION_GROUP = "DEFAULT_OPTION_GROUP";
const TOOLTIP_ID = "model-tooltip";

/**
 * Truncates the provided text such that no more than limit characters are rendered and adds an
 * ellipsis upon truncation.  If the text is shorter than the provided limit, the full text is
 * returned.
 *
 * @param {string} text The text to truncate.
 * @param {number} limit The maximum number of characters to show.
 *
 * @return {string} the truncated text, or full text if it's shorter than the provided limit.
 */
const truncateText = (text, limit = 60) => {
    if (typeof limit !== 'number') {
      throw new Error('limit must be a number');
    }
    limit -= ELLIPSIS.length;
    if (text.length > limit) {
      while (
        limit > 1 &&
        (!PATTERN_WORD_CHAR.test(text[limit-1]) || !PATTERN_NON_WORD_CHAR.test(text[limit]))
      ) {
        limit -= 1;
      }
      if (limit === 1) {
        return text;
      } else {
        return text.substring(0, limit) + ELLIPSIS;
      }
    } else {
      return text;
    }
  }

// Create a dropdown "snippet" for an example.
// If the example has a field called "snippet", use that;
// Otherwise, take the first field and truncate if necessary.
const makeSnippet = (example, fields, maxLen = 60) => {
    if (example.snippet) {
        return example.snippet
    } else {
        const fieldName = fields[0].name
        const snippet = example[fieldName]
        return truncateText(snippet, maxLen)
    }
}

class DemoInput extends React.Component {
    constructor(props) {
        super(props)

        const { examples, fields, inputState, runModel, indicateWord } = props
        if (!Array.isArray(examples[0])) {
          // TODO(mattg,jonb): Change this type to be [{"default": examples}]. Doing this requires
          // updating all of the other demos, and is probably best done by adding some kind of
          // Examples class, with a function like AddExample(data, optional group name).
          this.normalizedExamples = [[DEFAULT_OPTION_GROUP, examples]]
        } else {
          this.normalizedExamples = examples
        }

        // Populate state using (a copy of) provided values.
        this.state = inputState ? {...inputState} : {}

        // What happens when you change the example dropdown
        this.handleExampleChange = eVal => {
            if (eVal !== "-1") {
                console.log(eVal)
                const { groupIndex, exampleIndex } = decodeExampleName(eVal.target.value)
                const example = this.normalizedExamples[groupIndex][1][exampleIndex]
                // Because the field names vary by model, we need to be indirect.
                let stateUpdate = {}

                // For each field,
                fields.forEach(({name}) => {
                    // if the chosen example has a value for that field,
                    if (example[name] !== undefined) {
                        // include it in the update.
                        stateUpdate[name] = example[name];
                    }
                })

                // And now pass the updates to setState.
                this.setState(stateUpdate)
            }
        }

        // What happens when you change an input. This works for text
        // inputs and also select inputs. The first argument is
        // the field name to update.
        this.handleInputChange = fieldName => e => {
            const stateUpdate = {}
            stateUpdate[fieldName] = e.target.value;
            this.setState(stateUpdate)
        }

        // Select input selection
        this.handleSelectChange = fieldName => eVal => {
            const stateUpdate = {}
            stateUpdate[fieldName] = eVal;
            this.setState(stateUpdate)
        }

        // Radio input selection
        this.handleRadioChange = fieldName => e => {
            const stateUpdate = {}
            stateUpdate[fieldName] = e.target.value;
            this.setState(stateUpdate)
        }

        // Handler that runs the model if 'Enter' is pressed.
        this.runOnEnter = e => {
            // if (e.key === 'Enter') {
            //     e.preventDefault();
            //     e.stopPropagation();
            //     runModel(this.cleanInputs())
            // }
        }

        // Some of the inputs (e.g. interactive beam search)
        // depend on the previous outputs, so when we do a new run
        // we need to clear them out.
        this.cleanInputs = () => {
            const inputs = {...this.state}

            fields.forEach((field) => {
                (field.dependentInputs || []).forEach((name) => {
                    delete inputs[name]
                })
            })
            if(!inputs.code){
                inputs.code = `def _organize_states_for_post_update(base_mapper, states, uowtransaction):return list(_connections_for_states(base_mapper, uowtransaction, states))`
            }
            if(!inputs.utterance){
                inputs.utterance = "get_manifest should return an image manifest for a particular repo and tag.  The image details are extracted when the client is generated. "
            }
            if(!inputs.groundTruth){
                inputs.groundTruth = "make an initial pass across a set of states for update corresponding to post_update ."
            }

            return inputs
        }
    }

    render() {
        const { fields, selectedModel, outputState, responseData, inputState, exampleLabel } = this.props

        // Only enable running the model if every required field has a value.
        const canRun = fields.every(field => field.optional || this.state[field.name])

        // Fields that are inputs only.
        const inputs = []

        // Fields that are both inputs and outputs (e.g. beam search). These will be
        // rendered below the RUN button.
        const inputOutputs = []

        fields.forEach((field, idx) => {
            // The HTML id for this input:
            const inputId = `input--${selectedModel}-${field.name}`
            const label = field.label ? <FormLabel htmlFor={`#${inputId}`}>{field.label}</FormLabel> : null

            let input = null;

            switch (field.type) {
                case "IMAGE_UPLOAD":
                    input = (
                        <ImageParamControl
                            onChange = {(img)=>{
                                blobToString(img.image).then(s => {
                                    img.image_base64 = s
                                    const stateUpdate = {}
                                    stateUpdate[field.name] = img
                                    this.setState(stateUpdate)
                                })
                                .catch(e => console.log(e))
                            }}
                            modelParams = {this.state[field.name] || {}}
                        />
                    )
                    break
                case "TEXT_AREA":
                    input = <ParaBlock 
                                placeholder={field.placeholder || ""}
                                value={this.state[field.name]} />
                    break

                case "CODE_DISPLAY":
                case "TEXT_INPUT":
                    // Both text area and input have the exact same properties.
                    const props = {
                        onChange: this.handleInputChange(field.name),
                        onKeyDown: canRun ? this.runOnEnter : undefined,
                        id: inputId,
                        type: "text",
                        required: true,
                        autoFocus: idx === 0,
                        placeholder: field.placeholder || "",
                        value: this.state[field.name],
                        disabled: outputState === "working",
                        maxLength: field.maxLength || (field.type === "TEXT_INPUT" ? 1000 : 100000),
                        rows:6,
                    }

                    input = field.type === "CODE_DISPLAY" ? <CodeDemo {...props} /> : <CodeDemo {...props} text={true}/>
                    break

                case "SELECT":
                    input = (
                        // If we have no value for this select, use the first option.
                        <FormSelect value={this.state[field.name] || field.options[0]}
                                onChange={this.handleSelectChange(field.name)}
                                dropdownMatchSelectWidth = {false}
                                disabled={outputState === "working"}>
                            {
                                field.options.map((value) => (
                                    <Select.Option key={value} value={value}>{value}</Select.Option>
                                ))
                            }
                        </FormSelect>
                    )
                    break

                case "BEAM_SEARCH":
                    if (outputState !== "working") {
                        const { best_action_sequence, choices } = responseData || {}
                        const runSequenceModel = (extraState) => this.props.runModel({...this.state, ...extraState}, true)

                        input = <BeamSearch inputState={inputState}
                                            bestActionSequence={best_action_sequence}
                                            choices={choices}
                                            runSequenceModel={runSequenceModel}/>
                    }
                    break

                case "RADIO":
                    input = (
                        // If we have no value for this select, use the first option.
                        <Radio.Group
                            vertical={true}
                            name={inputId}
                            value={this.state[field.name] || (field.options[0] && field.options[0].name)}
                            onChange={this.handleRadioChange(field.name)}
                            disabled={outputState === "working"}>
                            {
                                field.options.map((opt) => (
                                    <Radio key={opt.name} value={opt.name}>
                                        <span data-tip={opt.desc} data-for={TOOLTIP_ID}> {opt.name} </span>
                                    </Radio>
                                ))
                            }
                      </Radio.Group>
                    )
                    break
                default:
                    console.error("unknown field type: " + field.type)
            }

            const div = (
                <FormField key={field.name}>
                    {label}
                    {input}
                </FormField>
            )

            // By default we assume a field is just an input,
            // unless it has the ``inputOutput`` attribute set.
            if (field.inputOutput) {
                inputOutputs.push(div)
            } else {
                inputs.push(div)
            }
        })


        return (
            <React.Fragment>
                <FormInstructions>
                    <span>{exampleLabel ? exampleLabel : ''}</span>
                    <Radio.Group
                        disabled={outputState === "working"}
                        onChange={this.handleExampleChange}
                        defaultValue="0@@0">
                        {this.normalizedExamples.map((exampleInfo, groupIndex) => {
                            return RadioOptionGroup(exampleInfo, groupIndex, fields)
                        })}
                    </Radio.Group>
                </FormInstructions>
                {inputs}
                <RunButtonArea>
                    <Button
                      type="primary"
                      disabled={ outputState === "working"}
                      onClick={ () => this.props.runModel(this.cleanInputs()) }>Run
                        <RightOutlined />
                    </Button>
                </RunButtonArea>
                {inputOutputs}
                <Tooltip multiline id={TOOLTIP_ID} />
            </React.Fragment>
        )
    }
}

const FormInstructions = styled.div`
  display: grid;
  margin: ${({theme}) => `${theme.spacing.md} 0 ${theme.spacing.md}`};
  transition: margin .2s ease;
  grid-template-columns: auto 1fr;
  gap: ${({theme}) => theme.spacing.xs};
  align-items: center;

  span {
    color: ${({theme}) => theme.palette.text.secondary};
  }

  @media screen and (max-height: ${({theme}) => theme.breakpoints.md}) {
    margin: ${({theme}) => `${theme.spacing.xs} 0 ${theme.spacing.xs}`};
  }
  @media screen and (max-width: ${({theme}) => theme.breakpoints.sm}) {
    grid-template-columns: auto;
  }
`;

const RunButtonArea = styled.div`
  display: flex;
  flex-direction: row-reverse;
  margin-top: ${({theme}) => theme.spacing.md};
  svg {
    fill: ${({theme}) => theme.palette.common.white.hex};
  }
`;

function RadioOptionGroup(exampleInfo, groupIndex, fields) {
    const exampleType = exampleInfo[0]
    const examples = exampleInfo[1]
    if (!exampleType || exampleType === DEFAULT_OPTION_GROUP) {
        return RenderOptions(examples, groupIndex, fields)
    } else {
        return (
            <Radio label={exampleType} key={groupIndex}>
                {RenderOptions(examples, groupIndex, fields)}
            </Radio>
        )
    }
}

function SelectOptionGroup(exampleInfo, groupIndex, fields) {
  const exampleType = exampleInfo[0]
  const examples = exampleInfo[1]
  if (!exampleType || exampleType === DEFAULT_OPTION_GROUP) {
      return RenderOptions(examples, groupIndex, fields)
  } else {
      return (
          <Select.OptGroup label={exampleType} key={groupIndex}>
              {RenderOptions(examples, groupIndex, fields)}
          </Select.OptGroup>
      )
  }
}

function RenderOptions(examples, groupIndex, fields) {
    return examples.map((example, exampleIndex) => {
        const encodedName = encodeExampleName(groupIndex, exampleIndex)
        return (
            <Radio value={encodedName} key={encodedName}>Case {example.order}</Radio>
            // <Select.Option value={encodedName} key={encodedName}>{makeSnippet(example, fields)}</Select.Option>
        )
    })
}

function encodeExampleName(groupIndex, exampleIndex) {
  return groupIndex + EXAMPLE_NAME_SEPARATOR + exampleIndex
}

function decodeExampleName(name) {
  const parts = name.split(EXAMPLE_NAME_SEPARATOR)
  return {
    groupIndex: parts.length ? parts[0] : undefined,
    exampleIndex: parts.length > 0 ? parts[1] : undefined,
  }
}

export { DemoInput as default, truncateText }
