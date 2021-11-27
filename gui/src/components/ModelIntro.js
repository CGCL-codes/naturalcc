import React from 'react';

/*******************************************************************************
  <ModelIntro /> Component
*******************************************************************************/

class ModelIntro extends React.Component {
  constructor(props){
    super(props);

    this.state = {
      showFullDescription: false
    }
  }

  toggleShowMore() {
    this.setState({showFullDescription: !this.state.showFullDescription})
  }

  render() {

    const { title, description } = this.props;

    return (
      <div>
        <h2><span>{title}</span></h2>
        <p>
          <span>{description}</span>
        </p>
      </div>
    );
  }
}

export default ModelIntro;
