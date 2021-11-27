import React from 'react';
import styled from 'styled-components';

import '../css/Pane.css';
import '../css/model.css';
import '../css/passage.css';

/*******************************************************************************
  <ResultDisplay /> Component
*******************************************************************************/

class ResultDisplay extends React.Component {

    render() {
      const { resultPane, outputState } = this.props;

      const placeholderTemplate = (message) => {
        return (
          <div className="placeholder">
            <div className="placeholder__content">
              <svg className={`placeholder__${outputState}`}>
                <use xlinkHref={`#icon__${outputState}`}></use>
              </svg>
              {message !== "" ? (
                <p>{message}</p>
              ) : null}
            </div>
          </div>
        );
      }

      let outputContent;
      switch (outputState) {
        case "working":
          outputContent = placeholderTemplate("");
          break;
        case "received":
          outputContent = this.props.children;
          break;
        case "error":
          outputContent = placeholderTemplate("Something went wrong. Please try again.");
          break;
        default:
          // outputState = "empty"
          outputContent = placeholderTemplate("Run model to view results");
      }

      return (
        <div className={`pane__${resultPane} model__output ${outputState !== "received" ? "model__output--empty" : ""}`}>
          <div className="pane__thumb"></div>
          {outputContent}
        </div>
      );
    }
}


/*******************************************************************************
  <PaneRight /> Component
*******************************************************************************/

export class PaneRight extends React.Component {
    render() {
      const { outputState } = this.props;

      return (
        <ResultDisplay resultPane="right" outputState={outputState}>
          {this.props.children}
        </ResultDisplay>
      )
    }
}

/*******************************************************************************
  <PaneBottom /> Component
*******************************************************************************/

export class PaneBottomBase extends React.Component {
  render() {
    const { outputState } = this.props;

    return (
      <ResultDisplay className={this.props.className} resultPane="bottom" outputState={outputState}>
        {this.props.children}
      </ResultDisplay>
    )
  }
}

export const PaneBottom = styled(PaneBottomBase)`
  background: ${({theme}) => theme.palette.common.white.hex};
`;


/*******************************************************************************
<PaneLeft /> Component
*******************************************************************************/

export class PaneLeft extends React.Component {

    render () {
      return (
        <div className="pane__left">
          {this.props.children}
        </div>
      );
    }
}

/*******************************************************************************
<PaneTop /> Component
*******************************************************************************/

class PaneTopBase extends React.Component {

  render () {
    return (
      <div className={this.props.className}>
        {this.props.children}
      </ div>
    );
  }
}

export const PaneTop = styled(PaneTopBase)`
  background-color: ${({theme}) => theme.palette.common.white.hex};
  width: 100%;
  align-self: stretch;
  display: block;
`;
