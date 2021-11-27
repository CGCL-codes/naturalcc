import React from 'react';

import { UsageSection } from './UsageSection';
import { UsageHeader } from './UsageHeader';
import { UsageCode } from './UsageCode';
import SyntaxHighlight from './highlight/SyntaxHighlight';

export const Usage = (usage) => ( // usage: {installNote?: Element, installCommand?: string, bashNote?: Element, bashCommand?: string, pythonNote?: Element, pythonCommand?: string, predictorNote?: Element, predictorCommand?: string, evaluationNote?: Element, evaluationCommand?: string}
    <React.Fragment>
        <UsageSection>
            <UsageHeader>Installing AllenNLP</UsageHeader>
            {usage.installNote ? <p>{usage.installNote}</p> : null}
            {usage.installCommand
                ? <UsageCode>
                    <SyntaxHighlight language="bash">{usage.installCommand}</SyntaxHighlight>
                </UsageCode>
                : null}
            <UsageHeader>Prediction</UsageHeader>
            <strong>On the command line (bash):</strong>
            {usage.bashNote ? <p>{usage.bashNote}</p> : null}
            {usage.bashCommand
                ? <UsageCode>
                    <SyntaxHighlight language="bash">{usage.bashCommand}</SyntaxHighlight>
                </UsageCode>
                : null}
            <strong>As a library (Python):</strong>
            {usage.pythonNote ? <p>{usage.pythonNote}</p> : null}
            {usage.pythonCommand
                ? <UsageCode>
                    <SyntaxHighlight language="python">{usage.pythonCommand}</SyntaxHighlight>
                </UsageCode>
                : null}
        </UsageSection>
        <UsageSection>
            <UsageHeader>Evaluation</UsageHeader>
            {usage.evaluationNote ? <p>{usage.evaluationNote}</p> : null}
            {usage.evaluationCommand
                ? <UsageCode>
                    <SyntaxHighlight language="python">{usage.evaluationCommand}</SyntaxHighlight>
                </UsageCode>
                : null}
        </UsageSection>
        <UsageSection>
            <UsageHeader>Training</UsageHeader>
            {usage.trainingNote ? <p>{usage.trainingNote}</p> : null}
            {usage.trainingCommand
                ? <UsageCode>
                    <SyntaxHighlight language="python">{usage.trainingCommand}</SyntaxHighlight>
                </UsageCode>
                : null}
        </UsageSection>
    </React.Fragment>
)
