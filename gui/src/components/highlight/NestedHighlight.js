import React from 'react';
import { Highlight } from './Highlight';
import HighlightContainer from './HighlightContainer';

const highlightColors = [
  "blue",
  "green",
  "pink",
  "orange",
  "purple",
  "teal",
  "tan",
  "red",
  "cobalt",
  "brown",
  "slate",
  "fuchsia",
  "gray"
];

/**
 * Matches an index with a color. If index is greater than number of colors, cycle through colors.
 * @param {number} index
 */
export const getHighlightColor = (index) => {
  if (index <= highlightColors.length) {
    return highlightColors[index];
  } else {
    return highlightColors[index - (highlightColors.length * Math.floor(index / highlightColors.length))];
  }
}

/**
 * Helper function for transforming response data into a tree object.
 *
 * @param {string[]} tokens a list of strings of each of the tokens (words or punctuation) present
 * @param {{ labelString: number[][] } | number[][][]} clusters a collection of arrays that specify spans to be clustered in the original list of tokens
 */
const transformToTree = (tokens, clusters) => {
  // Span tree data transform code courtesy of Michael S.
  function contains(span, index) {
    return index >= span[0] && index <= span[1];
  }

  let insideClusters = [
    {
      cluster: -1,
      contents: [],
      end: -1
    }
  ];

  tokens.forEach((token, i) => {
    // Find all the new clusters we are entering at the current index
    let newClusters = [];
    Object.keys(clusters).forEach((key, j) => {
      const cluster = clusters[key];
      // Make sure we're not already in this cluster
      if (!insideClusters.map((c) => c.cluster).includes(key)) {
        cluster.forEach((span) => {
          if (contains(span, i)) {
              newClusters.push({ end: span[1], cluster: key, clusterIndex: j });
          }
        });
      }
    });

    // Enter each new cluster, starting with the leftmost
    newClusters.sort(function(a, b) { return b.end - a.end }).forEach((newCluster) => {
      // Descend into the new cluster
      insideClusters.push(
        {
          cluster: newCluster.cluster,
          contents: [],
          end: newCluster.end,
          clusterIndex: newCluster.clusterIndex
        }
      );
    });

    // Add the current token into the current cluster
    insideClusters[insideClusters.length-1].contents.push(token);

    // Exit each cluster we're at the end of
    while (insideClusters.length > 0 && insideClusters[insideClusters.length-1].end === i) {
      const topCluster = insideClusters.pop();
      insideClusters[insideClusters.length-1].contents.push(topCluster);
    }
  });

  return insideClusters[0].contents;
}

/**
 * Not meant to be used outside of the Nested Highlight.
 */
const InnerHighlight = props => {
  const {
    activeDepths,
    activeIds,
    data,
    depth,
    isClickable,
    isClicking,
    labelPosition,
    onMouseDown,
    onMouseOut,
    onMouseOver,
    onMouseUp,
    selectedId,
    highlightColor,
    tokenSeparator
  } = props;
  const lenData = data.length;
  return (
      data.map((token, idx) => {
        return typeof(token) === "object" && !(React.isValidElement(token)) ? (
          <Highlight
            activeDepths={activeDepths}
            activeIds={activeIds}
            color={
              (typeof highlightColor === 'function'
                ? highlightColor(token)
                : highlightColor) ||
              getHighlightColor(token.clusterIndex)}
            depth={depth}
            id={token.cluster}
            isClickable={isClickable}
            isClicking={isClicking}
            key={idx}
            label={token.cluster}
            labelPosition={labelPosition}
            onMouseDown={onMouseDown}
            onMouseOut={onMouseOut}
            onMouseOver={onMouseOver}
            onMouseUp={onMouseUp}
            selectedId={selectedId}
          >
              <InnerHighlight
                activeDepths={activeDepths}
                activeIds={activeIds}
                data={token.contents}
                depth={depth + 1}
                isClickable={isClickable}
                isClicking={isClicking}
                labelPosition={labelPosition}
                onMouseDown={onMouseDown}
                onMouseOut={onMouseOut}
                onMouseOver={onMouseOver}
                onMouseUp={onMouseUp}
                selectedId={selectedId}
              />
          </Highlight>
        ) : (
          <span key={idx}>{token}{idx === lenData - 1 ? null : tokenSeparator || <>&nbsp;</>}</span>
        )
      })
  )
}

/**
 * A function that recursively handles rendering spans of text to highlight.
 * Use in conjunction withHighlightClickHandling if isClickable is true.
 * @param {
 *  activeDepths?: { ids: string[], depths: number[]}
 *  activeIds?: string[]
 *  clusters: number[][][] | { [id: string]: number[][] }
 *  isClickable?: boolean
 *  isClicking?: boolean
 *  labelPosition: "top" | "left" | "right" | "bottom"
 *  onMouseDown?: (id: string, depth: number) => void
 *  onMouseOut?: (id: string) => void
 *  onMouseOver?: (id: string) => void
 *  onMouseUp?: (id: string) => void
 *  selectedId?: string
 *  tokens: string[],
 *  highlightColor?: string|(token: object) => string;
 *  tokenSeparator?: string;
 * } props
 */
const NestedHighlight = props => {
  const {
    activeDepths,
    activeIds,
    clusters,
    isClickable,
    isClicking,
    labelPosition,
    onMouseDown,
    onMouseOut,
    onMouseOver,
    onMouseUp,
    selectedId,
    tokens,
    highlightColor,
    tokenSeparator
  } = props;
  const data = transformToTree(tokens, clusters);
  return (
    <HighlightContainer>
      <InnerHighlight
        activeDepths={activeDepths}
        activeIds={activeIds}
        data={data}
        depth={0}
        isClickable={isClickable}
        isClicking={isClicking}
        labelPosition={labelPosition}
        onMouseDown={onMouseDown}
        onMouseOut={onMouseOut}
        onMouseOver={onMouseOver}
        onMouseUp={onMouseUp}
        selectedId={selectedId}
        highlightColor={highlightColor}
        tokenSeparator={tokenSeparator}
      />
    </HighlightContainer>
  );
}
export default NestedHighlight;

/**
 * An HOC that handles highlight click handling state and passes through
 * any given props.
 *
 * @param {React.Component} WrappedComponent Any component that requires highlight click handling
 */
export const withHighlightClickHandling = WrappedComponent => {
  return class _withHighlightClickHandling extends React.Component {
    state = {
      selectedCluster: -1,
      activeIds: [],
      activeDepths: {ids:[],depths:[]},
      selectedId: null,
      isClicking: false
    };

    handleHighlightMouseDown = (id, depth) => {
      let depthTable = this.state.activeDepths;
      depthTable.ids.push(id);
      depthTable.depths.push(depth);

      this.setState({
        activeIds: [id],
        activeDepths: depthTable,
        isClicking: true
      });
    }

    handleHighlightMouseUp = (id) => {
      const depthTable = this.state.activeDepths;
      const deepestIndex = depthTable.depths.indexOf(Math.max(...depthTable.depths));

      this.setState(prevState => ({
        selectedId: prevState.selectedId === depthTable.ids[deepestIndex] ? null : depthTable.ids[deepestIndex],
        isClicking: false,
        activeDepths: {ids:[],depths:[]},
        activeIds: [...prevState.activeIds, id],
      }));
    }

    handleHighlightMouseOver = (id) => {
      this.setState(prevState => ({
        activeIds: [...prevState.activeIds, id],
      }));
    }

    handleHighlightMouseOut = () => {
      this.setState(prevState => ({
        activeIds: prevState.activeIds.filter(i => (i === this.state.selectedId)),
      }));
    }

    render() {
      const { activeIds, activeDepths, isClicking, selectedId } = this.state;
      return (
        <WrappedComponent
          activeDepths={activeDepths}
          activeIds={activeIds}
          isClicking={isClicking}
          onMouseDown={this.handleHighlightMouseDown}
          onMouseOut={this.handleHighlightMouseOut}
          onMouseOver={this.handleHighlightMouseOver}
          onMouseUp={this.handleHighlightMouseUp}
          selectedId={selectedId}
          {...this.props}
        />
      )
    }
  }
}
