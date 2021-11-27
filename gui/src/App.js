import React from 'react';
import styled from 'styled-components';
import {BrowserRouter as Router, Route, Redirect, Switch} from 'react-router-dom';
import { ThemeProvider } from '@allenai/varnish'
import { Layout} from 'antd';
import Menu from './components/Menu';
import {ScrollToTopOnPageChange} from './components/ScrollToTopOnPageChange';
import {modelComponents, modelRedirects} from './models'

import WaitingForPermalink from './components/WaitingForPermalink';

import './css/App.css';
import '@allenai/varnish/dist/varnish.css';
// const { HeaderColumns, HeaderTitle } = Header;
// const {HeaderColumns} = Header;
const { Header, Sider, Content} = Layout

const DEFAULT_PATH = "/code-summarization"

const App = () => (
    <ThemeProvider>
        <Router>
            <ScrollToTopOnPageChange/>
            <Switch>
                <Route exact path="/" render={() => (
                    <Redirect to={DEFAULT_PATH}/>
                )}/>
                <Route path="/:model/:slug?" component={Demo}/>
            </Switch>
        </Router>
    </ThemeProvider>
)

const Demo = (props) => {
    const {model, slug} = props.match.params
    const redirectedModel = modelRedirects[model] || model
    return (
        <Layout>
            <Header style={{zIndex:"999"}}>
                <p style={{color:"white",fontWeight:"bolder", fontSize:"1.5rem"}}>Natural CC</p>
            </Header>
            <Layout theme="light">
                <Menu redirectedModel={redirectedModel}/>
                <Layout>
                    <div style={styles.gitForkTag}>
                        <a style={styles.gitLink} href="https://github.com/CGCL-codes/naturalcc-demo" target="_blank">Fork on Github</a>
                    </div>
                    <SingleTaskDemo model={redirectedModel} slug={slug}/>
                    <div style={styles.acknowledgement}>
                        <i>Created by <a href="https://ant-design.gitee.io/" target="_blank">ant-design</a></i>
                    </div> 
                </Layout>
            </Layout>
        </Layout>
    );
}
const styles = {
    gitForkTag: {
        position: "absolute",
        top:"0",
        right: "0",
        backgroundColor: "#f9f9f9",
        borderTop: "1px solid #e8e8e8",
        borderBottom:"1px solid #e8e8e8",
        padding: "5px 0px",
        transform: "rotate(45deg) translateY(58px) translateX(74px)",
        zindex: "1"
      },
    gitLink: {
        textDecoration: "none",
        color: "black"
    },
    acknowledgement: {
        position: "fixed",
        bottom: "1.5rem",
        right: "1rem",
        color: "grey"
    }
}

const FullSizeContent = styled(Content)`
    padding: 0;
`;

// const HeaderColumnsWithSpace = styled(HeaderColumns)`
//     padding: ${({theme}) => theme.spacing.md} 0;
// `;

class SingleTaskDemo extends React.Component {
    constructor(props) {
        super(props);

        // React router supplies us with a model name and (possibly) a slug.
        const {model, slug} = props;

        this.state = {
            slug,
            selectedModel: model,
            requestData: null,
            responseData: null
        }
    }

    // We also need to update the state whenever we receive new props from React router.
    componentDidUpdate() {
        const {model, slug} = this.props;
        if (model !== this.state.selectedModel || slug !== this.state.slug) {
            const isModelChange = model !== this.state.selectedModel;
            const responseData = (
                isModelChange
                    ? null
                    : this.state.responseData
            );
            const requestData = (
                isModelChange
                    ? null
                    : this.state.requestData
            );
            this.setState({selectedModel: model, slug, responseData, requestData});
        }
    }

    // After the component mounts, we check if we need to fetch the data
    // for a permalink.
    componentDidMount() {
        const {slug, responseData} = this.state;

        // If this is a permalink and we don't yet have the data for it...
        // if (slug && !responseData) {
        //     // Make an ajax call to get the permadata,
        //     // and then use it to update the state.
        //     fetch(`/api/permalink/${slug}`)
        //         .then((response) => {
        //             return response.json();
        //         }).then((json) => {
        //         const {request_data} = json;
        //         this.setState({requestData: request_data});
        //     }).catch((error) => {
        //         // If a permalink doesn't resolve, we don't want to fail. Instead remove the slug from
        //         // the URL. This lets the user at least prepare a submission.
        //         console.error('Error loading permalink:', error);
        //         // Start over without the slug.
        //         window.location.replace(window.location.pathname.replace(`/${slug}`, ''));
        //     });
        // }
    }

    render() {
        const {slug, selectedModel, requestData, responseData} = this.state;
        console.log(slug, selectedModel, requestData, responseData);
        const updateData = (requestData, responseData) => this.setState({requestData, responseData})

        if (slug && !requestData) {
            // We're still waiting for permalink data, so just return the placeholder component.
            return (<WaitingForPermalink/>);
        } else if (modelComponents[selectedModel]) {
            // This is a model we know the component for, so render it.
            return React.createElement(modelComponents[selectedModel], {
                requestData,
                responseData,
                selectedModel,
                updateData
            })
            // return (<WaitingForPermalink/>);
        }
    }
}

export default App;
