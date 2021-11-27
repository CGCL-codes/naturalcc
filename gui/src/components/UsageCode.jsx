import styled from 'styled-components';

/**
 * Create a little padding and border around code samples in the
 * usage sections, for readability and spacing.
 */
export const UsageCode = styled.div`
    padding: ${({ theme }) => theme.spacing.sm};
    border: 1px solid ${({ theme }) => theme.palette.border.main};
    margin: ${({ theme }) => `${theme.spacing.sm} 0 ${theme.spacing.md}`};

    pre {
        margin: 0;
    }
`;
