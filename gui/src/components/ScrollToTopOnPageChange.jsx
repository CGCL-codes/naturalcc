import { useEffect } from 'react';
import { withRouter } from 'react-router';

/**
 * Use this component inside a top-level <Route /> handler when you'd like
 * the page to be scrolled to the top after a URL change.
 */
const ScrollToTopOnPageChangeImpl = ({ history }) => {
    useEffect(() =>
        history.listen(() => {
            window.scrollTo(0, 0);
        })
    );
    return null;
};

export const ScrollToTopOnPageChange = withRouter(ScrollToTopOnPageChangeImpl);
