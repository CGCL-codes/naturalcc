import styled from 'styled-components';
import { NavLink } from 'react-router-dom';
import { Link } from '@allenai/varnish';

const { linkColorStyles } = Link

export const InternalLink = styled(NavLink)`
    ${linkColorStyles()}
`;
