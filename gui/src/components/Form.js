import styled, { css } from 'styled-components';
import { Input, Select } from '@allenai/varnish';

export const FormField = styled.div`
  margin-top: ${({theme}) => theme.spacing.md};
  transition: margin .2s ease;

  @media (max-height: ${({theme}) => theme.breakpoints.md}) {
    margin-top: ${({theme}) => theme.spacing.xs};
  }
`;

export const FormLabel = styled.label`
  display: block;
  font-weight: ${({theme}) => theme.typography.bodyBold.fontWeight};
  margin-top: ${({theme}) => theme.spacing.xs};
  transition: font-size .2s ease;
`;

const baseInputStyles = css`
  width: 100%;
  margin-top: ${({theme}) => theme.spacing.xs};
  display: block;
  transition: min-height .2s ease, opacity .2s ease;

  &:focus {
    outline: 0;
    box-shadow: 0 0 ${({theme}) => theme.spacing.md} ${({theme}) => theme.palette.primary};
  }

  @media (max-height: ${({theme}) => theme.breakpoints.md}) {
    margin-top: ${({theme}) => theme.spacing.xxs};
  }
`;

export const FormTextArea = styled(Input.TextArea)`
  && {
    ${baseInputStyles}

    resize: vertical;
    min-height: 5.4em;

    @media (max-height: ${({theme}) => theme.breakpoints.md}) {
      min-height: 4.4em;
    }
  }
`;

export const FormInput = styled(Input)`
  ${baseInputStyles}
`;

export const FormSelect = styled(Select)`
  ${baseInputStyles}
`;
