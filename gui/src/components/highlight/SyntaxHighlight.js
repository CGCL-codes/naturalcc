import React from 'react';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { vs } from 'react-syntax-highlighter/dist/esm/styles/hljs';

/*******************************************************************************
  <SyntaxHighlight /> Component

  This component is a wrapper for the `react-syntax-highlighter` component.
  The goal was to keep the highlight styling consistent and as easy to leverage
  as possible. This takes the style-related props out of the equation for
  general use throughout AllenNLP demos. Global code style can be managed here.

  Documentation for `react-syntax-highlighter` can be found on NPM:
  https://www.npmjs.com/package/react-syntax-highlighter

  Supported Languages:
  https://github.com/conorhastings/react-syntax-highlighter/blob/HEAD/AVAILABLE_LANGUAGES_HLJS.MD

  Supported Styles:
  https://github.com/conorhastings/react-syntax-highlighter/blob/HEAD/AVAILABLE_STYLES_HLJS.MD

  Demo:
  https://highlightjs.org/static/demo/

*******************************************************************************/

export default class SyntaxHighlight extends React.Component {
  render() {
    const {
      language, // string (optional, will auto-detct if not set explicitly)
      children  // string | object
    } = this.props;

    // Overriding the unwanted inline styles that `react-syntax-highlighter` adds by default:
    const customStyle = {
      background: 'transparent',
      padding: '0'
    };

    return (
      <SyntaxHighlighter
        language={language}
        style={vs}
        customStyle={customStyle}>
        {children}
      </SyntaxHighlighter>
    );
  }
}
