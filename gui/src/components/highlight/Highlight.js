import React from 'react';
import '../../css/Highlight.css';

/*******************************************************************************
  <Highlight /> Component
*******************************************************************************/

export const getHighlightConditionalClasses = (conditions) => {
  const {
    labelPosition,
    label,
    color,
    isClickable,
    selectedId,
    isClicking,
    id,
    activeDepths,
    deepestIndex,
    activeIds,
    children,
  } = conditions;
  return `highlight
    ${labelPosition ? labelPosition : label ? "bottom" : ""}
    ${color ? color : "blue"}
    ${isClickable ? "clickable" : ""}
    ${selectedId && selectedId === id ? "selected" : ""}
    ${isClicking && activeDepths.ids[deepestIndex] === id ? "clicking active" : ""}
    ${!isClicking && activeIds && activeIds.includes(id) ? "active" : ""}
    ${typeof(children) === "string" && children.length < 8 ? "short-text" : ""}`;
}

export const Highlight = props => {
  const {         // All fields optional:
    activeDepths,   // object
    activeIds,      // string[] | number[]
    children,       // object | string
    color,          // string (see highlightColors above for supported values)
    depth,          // number
    id,             // string | number
    isClickable,    // boolean
    isClicking,     // boolean
    label,          // string
    labelPosition,  // string (supported values: "top", "left", "right", "bottom")
    onClick,        // function
    onMouseDown,    // function
    onMouseOver,    // function
    onMouseOut,     // function
    onMouseUp,      // function
    selectedId,     // string | number
    secondaryLabel, // string
    tooltip         // string
  } = props;

  const deepestIndex = activeDepths ? activeDepths.depths.indexOf(Math.max(...activeDepths.depths)) : null;
  const conditionalClasses = getHighlightConditionalClasses({
    labelPosition,
    label,
    color,
    isClickable,
    selectedId,
    isClicking,
    id,
    activeDepths,
    deepestIndex,
    activeIds,
    children,
  });

  const labelTemplate = (
    <span className="highlight__label">
      <strong>{label}</strong>
      {secondaryLabel ? (
        <span className="highlight__label__secondary-label">{secondaryLabel}</span>
      ) : null}
    </span>
  );

  return (
    <span
      className={conditionalClasses}
      data-label={label}
      data-id={id}
      data-depth={depth}
      onClick={onClick ? () => { onClick(id) } : null}
      onMouseDown={onMouseDown ? () => { onMouseDown(id, depth) } : null}
      onMouseOver={onMouseOver ? () => { onMouseOver(id) } : null}
      onMouseOut={onMouseOut ? () => { onMouseOut(id) } : null}
      onMouseUp={onMouseUp ? () => { onMouseUp(id) } : null}>
      {(labelPosition === "left" || labelPosition === "top") ? labelTemplate : null}
      {children ? (
        <span className="highlight__content">{children}</span>
      ) : null}
      {(label || label !== null) && (labelPosition !== "left" && labelPosition !== "top") ? labelTemplate : null}
      {tooltip ? (
        <span className="highlight__tooltip">{tooltip}</span>
      ) : null}
    </span>
  );
}
