import React from 'react';
import '../../css/HighlightContainer.css';

/*******************************************************************************
  <HighlightContainer /> Component

  This is a Wrapper for <Highlight /> that sets
  container CSS classes that get inherited.
*******************************************************************************/

export default class HighlightContainer extends React.Component {
  render() {
    const {           // All fields optional:
      children,       // object | string
      isClicking,     // boolean
      layout,         // string (supported values: "bottom-labels", null)
      className
    } = this.props;

    const containerClasses = `passage
      model__content__summary
      highlight-container
      ${layout ? "highlight-container--" + layout : ""}
      ${isClicking ? "clicking" : ""}
      ${className || ""}`;

    return (
      <div className={containerClasses}>
        {children}
      </div>
    );
  }
}
