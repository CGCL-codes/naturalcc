const vscode = require('vscode');

function activate(context) {
  console.log('Python type hints plugin activated');

  let provider = vscode.languages.registerCompletionItemProvider(
    { language: 'python' },
    {
      provideCompletionItems(document, position) {
        const line = document.lineAt(position).text;
        const prefix = line.slice(0, position.character);

        if (!prefix.endsWith('->')) {
          return undefined;
        }

        // Define types and their corresponding inference processes
        const typeSuggestions = [
          {
            name: 'HttpResponse',
            inference:  'if_analysis(val, None, =):\n    T1 = return_analysis(HttpResponseNot\nFound(\'Content of id {id[:24]} not found.\nExpired?\'))\nT2 = return_analysis(res)\ncombine(T1,T2)'
          },
          {
            name: 'HttpResponseNotFound',
            inference: 'if_analysis(val, None, =):\n    T1 = return_analysis(HttpResponseNotFound("..."))\nT2 = return_analysis(res)\ncombine(T1,T2)'
          },
          {
            name: 'str',
            inference: 'if_analysis(val, None, =):\n    T1 = return_analysis(HttpResponseNotFound("..."))\nT2 = return_analysis(res)\ncombine(T1,T2)'
          },
          {
            name: 'Optional[str]',
            inference: 'if_analysis(val, None, =):\n    T1 = return_analysis(HttpResponseNotFound("..."))\nT2 = return_analysis(res)\ncombine(T1,T2)'
          },
          {
            name: 'bytes',
            inference: 'if_analysis(val, None, =):\n    T1 = return_analysis(HttpResponseNotFound("..."))\nT2 = return_analysis(res)\ncombine(T1,T2)'
          }
        ];

        return typeSuggestions.map((type, index) => {
          const item = new vscode.CompletionItem(type.name, vscode.CompletionItemKind.TypeParameter);
          item.insertText = type.name;
          item.detail = '💡 Recommended type';
          
          // Create hover documentation
          const doc = new vscode.MarkdownString();
          doc.appendMarkdown(`✨ **Expert suggestion**: Use \`${type.name}\` as return type\n\n`);
          doc.appendMarkdown(`### Type inference process\n\`\`\`\n${type.inference}\n\`\`\``);
          
          item.documentation = doc;
          item.sortText = '00' + index;
          return item;
        });
      }
    },
    '>'
  );

  context.subscriptions.push(provider);
}

function deactivate() {}

module.exports = {
  activate,
  deactivate
};