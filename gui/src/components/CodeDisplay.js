//import showndown from 'showdown'
//import showndownHighlight from 'showdown-highlight'
import React from 'react'
import 'prismjs/themes/prism.css'
const Prism = require('prismjs');
require('prismjs/components/prism-python.js');
class CodeDemo extends React.Component{
    render() {
        const { value, placeholder } = this.props
        console.log(this.props)
        const code = value?value:placeholder
        let codeHtml = {
            // Prism.highlight(text, grammar, language)
            // text: 需要格式化的代码
            // grammar: 需要格式化代码的语法
            // language: 需要格式化代码表示的语言
            __html: Prism.highlight(code, Prism.languages.python, 'python')
        }
        return (
            <>
                <pre className="language-python line-numbers">
                    <code dangerouslySetInnerHTML={codeHtml}></code>
                </pre>
            </>
        )
    }
}
export default CodeDemo
