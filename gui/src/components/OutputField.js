import React from 'react'

import { FormField, FormLabel } from './Form';

// A labeled output field with children
const OutputField = ({label, classes, children, suppressSummary}) => {
    const summaryClass = (label && !suppressSummary) ? 'model__content__summary ' : ''
    const extraClasses = classes || ''
    const className = summaryClass + extraClasses

    return (
        <FormField>
            {label ? <FormLabel>{label}</FormLabel> : null}
            {className ? (
                <div className={className}>
                    {children}
                </div>
                ) : children}
        </FormField>
    )
}

export default OutputField
