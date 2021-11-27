import React, { Component } from 'react'

export default class ParaBlock extends Component {
    render() {
        const { value, placeholder } = this.props
        const output = value ? value : placeholder
        return (
            <div>
                <pre className="language-">
                    {output}
                </pre>
            </div>
        )
    }
}
