import React from 'react';
import ReactTooltip from 'react-tooltip';
import '../css/HeatMap.css';

/*******************************************************************************
  <HeatMap /> Component

  Properties:

    data: Array[number[]]   * Array of arrays (rows) of numbers (cols) where each number
                              corresponds to a table cell / heatmap intensity value
    colLabels: string[]     * List of table header labels describing each column
    rowLabels: string[]     * List of table header labels describing each row
    includeSlider: bool     * Whether to include a slider to filter out values below a threshold
    showAllCols: bool       * Whether to always show all columns with the slider
    color: string           * Heatmap color (optional, default = "blue", see supportedColors below)
    normalization: string   * Sets normalization type (optional). Supported types:

      "none" (default, use this if you already have normalized probability distributions),
      "log-global" (does a global softmax over the whole matrix to get a probability distribution),
      "log-per-row" (does a softmax per row to get a probability distribution),
      "log-per-row-with-zero" (does a softmax per row, with the addition of a 0 logit),
      "linear" (finds the max and min values in the matrix, does a linear interpolation between them)

*******************************************************************************/

export default class HeatMap extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      activeRow: null,
      activeCol: null,
      filterOpacity: 0,
      showRowAt: undefined,
      showColAt: undefined,
      opacity: undefined,
      minFilterOpacity: 0,
      maxFilterOpacity: 1
    };
  }

  handleMouseOver(rowIndex, colIndex) {
    this.setState({
      activeRow: rowIndex,
      activeCol: colIndex
    });
  }

  handleMouseOut() {
    this.setState({
      activeRow: null,
      activeCol: null
    });
  }

  componentDidMount() {
    this.setOpacity(this.props);
  }

  componentDidUpdate(newProps) {
    if(this.props.data !== newProps.data || this.props.normalization !== newProps.normalization) {
        this.setOpacity(newProps);
    }
  }

  setOpacity(newProps){
    // HeatMap opacity via normalization conditional logic
    let opacity;
    if (newProps.normalization === "log-global") {
      const exped = newProps.data.map((x_list) => x_list.map((x) => Math.exp(x)));
      const flatArray = exped.reduce((i, o) => [...o, ...i], []);
      const sum = flatArray.reduce((a, b) => a + b, 0);
      opacity = exped.map((x_list) => x_list.map((x) => x / sum));
    } else if (newProps.normalization === "log-per-row") {
      const exped = newProps.data.map((x_list) => x_list.map((x) => Math.exp(x)));
      opacity = exped.map((x_list) => {
        const sum = x_list.reduce((a, b) => a + b, 0);
        return x_list.map((x) => x / sum);
      });
    } else if (newProps.normalization === "log-per-row-with-zero") {
      const exped = newProps.data.map((x_list) => x_list.map((x) => Math.exp(x)));
      opacity = exped.map((x_list) => {
        const sum = x_list.reduce((a, b) => a + b, 0) + Math.exp(0);
        return x_list.map((x) => x / sum);
      });
    } else if (newProps.normalization === "linear") {
      const flatArray = newProps.data.reduce((i, o) => [...o, ...i], []);
      const max = Math.max(...flatArray);
      const min = Math.min(...flatArray);
      if (max === min) {
        opacity = newProps.data;
      } else {
        opacity = newProps.data.map((x_list) => x_list.map((x) => ((x - min) / (max - min))));
      }
    } else {
      opacity = newProps.data;
    }

    const flatArray = opacity.reduce((i, o) => [...o, ...i], []);
    this.setState({opacity: opacity, minFilterOpacity: Math.min(...flatArray), maxFilterOpacity: Math.max(...flatArray)},
      () => this.setShowRowsAndColumns(this.state));
  }

  setShowRowsAndColumns(newState){
    if(newState.opacity && this.props.rowLabels && this.props.colLabels) {
      let showRowAt = this.props.rowLabels.map((label, index) => {
        return Math.max(...newState.opacity[index]) >= this.state.filterOpacity;
      });
      let transposeOpacity = newState.opacity[0].map((col, i) => newState.opacity.map(row => row[i]));
      let showColAt = this.props.colLabels.map((label, index) => {
        return Math.max(...transposeOpacity[index]) >= this.state.filterOpacity;
      });

      this.setState({ showRowAt, showColAt});
    }
  }

  render() {
    const { data,
            colLabels,
            rowLabels,
            color = "blue" } = this.props;

    const { activeRow,
            activeCol,
            showRowAt,
            showColAt,
            opacity } = this.state;

    const supportedColors = {
      // color values are [R,G,B]
      "red": [255,50,50],
      "green": [63, 201, 1],
      "blue": [50,159,255] // Default
    }

    if (!showRowAt || !showColAt || !opacity){
      return null; // loading
    }
    return (
      <div className="heatmap-container">
        {this.props.includeSlider && this.state.minFilterOpacity!==this.state.maxFilterOpacity &&
        <div className="slide_container">
          <input
            type="range"
            min={this.state.minFilterOpacity}
            max={this.state.maxFilterOpacity}
            step="0.001"
            value={this.state.filterOpacity}
            className="slider"
            onChange={e => this.setState({filterOpacity: Number(e.target.value)},
              () => this.setShowRowsAndColumns(this.state))} />
        </div>}
        <div className="heatmap-scroll">
          <div className="heatmap">
            <div className="heatmap__ft">
              <div className="heatmap__tr">
                <div className="heatmap__td heatmap__td--placeholder"></div>
                <div className="heatmap__td">
                  {/* BEGIN Column Labels */}
                  <table className="heatmap__col-labels">
                    <tbody>
                      <tr data-row="header">
                        {colLabels.map((colLabel, colIndex) => (
                          (this.props.showAllCols || this.state.showColAt[colIndex]) &&
                          <th className={`heatmap__label${colIndex === activeCol ? " heatmap__col-label-cursor" : ""}`}
                            key={`${colLabel}_${colIndex}`}
                            onMouseOver={() => this.handleMouseOver(null, colIndex)}
                            onMouseOut={() => this.handleMouseOut()}>
                            <div className="heatmap__label__outer">
                              <div className="heatmap__label__inner">
                                <span>{colLabel}</span>
                              </div>
                            </div>
                          </th>
                        ))}
                      </tr>
                    </tbody>
                  </table>{/* END Column Labels */}
                </div>{/* END .heatmap__td */}
              </div>{/* END .heatmap__tr */}
              <div className="heatmap__tr">
                <div className="heatmap__td">
                  {/* BEGIN Row Labels */}
                  <table className="heatmap__row-labels">
                    <tbody>
                      {rowLabels.map((rowLabel, rowIndex) => (
                        this.state.showRowAt[rowIndex] &&
                        <tr className="heatmap__row" key={`${rowLabel}_${rowIndex}`} data-row={rowIndex}>
                          <th
                            className={`heatmap__label${rowIndex === activeRow ? " heatmap__row-label-cursor" : ""}`}
                            onMouseOver={() => this.handleMouseOver(rowIndex, null)}
                            onMouseOut={() => this.handleMouseOut()}>
                            <div>{rowLabel}</div>
                          </th>
                        </tr>
                      ))}
                    </tbody>
                  </table>{/* END Row Labels */}
                </div>{/* END .heatmap__td */}
                <div className={`heatmap__td heatmap__datagrid-container heatmap__datagrid-container--${color}`}>
                  {/* BEGIN Data Grid */}
                  <table>
                    <tbody>
                      {rowLabels.map((rowLabel, rowIndex) => (
                        this.state.showRowAt[rowIndex] &&
                        <tr className="heatmap__row" key={`${rowLabel}_${rowIndex}`} data-row={rowIndex}>
                          {colLabels.map((colLabel, colIndex) => (
                            (this.props.showAllCols || this.state.showColAt[colIndex]) &&
                            <td key={`${colLabel}_${colIndex}_${rowLabel}_${rowIndex}`}
                              className="heatmap__cell"
                              style={{backgroundColor: `rgba(${supportedColors[color].join(",")},${opacity[rowIndex][colIndex]})`}}
                              data-tip=""
                              data-for="heatmap-tooltip">
                              {((rowIndex === activeRow && colIndex === activeCol) || (colIndex === activeCol && rowIndex === 0 && activeRow === null)) ? (
                                <div className="heatmap__col-cursor"></div>
                              ) : null}
                              {((rowIndex === activeRow && colIndex === activeCol) || (rowIndex === activeRow && colIndex === 0 && activeCol === null)) ? (
                                <div className="heatmap__row-cursor"></div>
                              ) : null}
                              <div className={`heatmap__trigger${rowIndex === activeRow && colIndex === activeCol ? " heatmap__cursor" : ""}`}
                                onMouseOver={() => this.handleMouseOver(rowIndex, colIndex)}
                                onMouseOut={() => this.handleMouseOut()}>
                              </div>
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>{/* END Data Grid */}
                </div>{/* END .heatmap__td */}
              </div>{/* END .heatmap__tr */}
            </div>{/* END .heatmap__ft */}
          </div>{/* END .heatmap */}
        </div>{/* END .heatmap-scroll */}
        {/* BEGIN Tooltip */}
        {activeRow !== null && activeCol !== null ? (
          <ReactTooltip
            id="heatmap-tooltip"
            className="heatmap-tooltip"
            place="right"
            effect="solid"
            delayHide={0}
            delayShow={0}
            delayUpdate={0}>
            {`${data[activeRow][activeCol]}`}
            <span className="heatmap-tooltip__meta"><strong>Row:</strong> {rowLabels[activeRow]}</span>
            <span className="heatmap-tooltip__meta"><strong>Column:</strong> {colLabels[activeCol]}</span>
          </ReactTooltip>
        ) : null /* END Tooltip */}
        {/* END .heatmap-container */}
      </div>
    );
  }
}
