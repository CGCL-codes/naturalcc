import styled from 'styled-components';
import ReactTooltip from 'react-tooltip';

export const Tooltip = styled(ReactTooltip)`
  && {
    &,
    span {
      text-rendering: geometricPrecision;
      font-size: ${({theme}) => theme.typography.bodySmall.fontSize};
      color: ${({theme}) => theme.typography.bodySmall.contrastColor};
      line-height:  ${({theme}) => theme.typography.bodySmall.lineHeight};
    }
  }
`;

export const ColorizedToken = styled.span`
  background-color: ${props => props.backgroundColor};
  padding: 1px;
  margin: 1px;
  display: inline-block;
  border-radius: 3px;
  font-weight: normal;
`;

// red token used to represent deletion in InputReduction and replacement in HotFlip
export const RedToken = styled.span`
  background-color: #FF5733;
  padding: 1px;
  margin: 1px;
  display: inline-block;
  border-radius: 3px;
  font-weight: normal;
`;

// green token used to represent addition in HotFlip
export const GreenToken = styled.span`
  background-color: #26BD19;
  padding: 1px;
  margin: 1px;
  display: inline-block;
  border-radius: 3px;
  font-weight: normal;
`;

// green token used to represent addition in HotFlip
export const TransparentToken = styled.span`
  background-color: "transparent";
  padding: 1px;
  margin: 1px;
  display: inline-block;
  border-radius: 3px;
  font-weight: normal;
`;

// all white (the UI doesn't display it) token used in InputReduction to show removal
export const BlankToken = styled.span`
  background-color: transparent;
  color: white;
  padding: 1px;
  margin: 1px;
  display: inline-block;
  border-radius: 3px;
`;