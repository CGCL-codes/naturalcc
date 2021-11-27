import styled from 'styled-components';

import { FormLabel } from './Form';

export const UsageHeader = styled(FormLabel)`
    margin: 0 0 ${({ theme }) => theme.spacing.xs};
`;
