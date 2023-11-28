import { getOverlap } from './kmp'
import {
    InlineCompletionItem,
    InlineCompletionItemProvider,
    InlineCompletionList,
    InlineCompletionContext,
    CancellationToken,
    Position,
    Range,
    TextDocument,
    workspace,
    window
  } from 'vscode'

export class NccCompletionProvider implements InlineCompletionItemProvider {

    private debounceTimer: NodeJS.Timeout | undefined;
    public async provideInlineCompletionItems(
        document: TextDocument,
        position: Position,
        context: InlineCompletionContext, 
        token: CancellationToken
      ): Promise<InlineCompletionItem[] | InlineCompletionList | null | undefined> {
        const line = document.lineAt(position.line).text

        if(line.length < 5) {
          return []
        }

        if (this.debounceTimer) {
          console.log('clear')
          clearTimeout(this.debounceTimer);
        }
        return new Promise((resolve) => {
          this.debounceTimer = setTimeout(async () => {
            console.log('trigger')
            const response = await fetch(encodeURI('http://127.0.0.1:5000/complete?prompt=' + line));
            const data = await response.json();
            const ret = data.generated_text;
            console.log(ret);
            const overlap = getOverlap(line, ret);
            resolve([new InlineCompletionItem(ret, new Range(position.translate(0, -overlap), position))]);
          }, 2000);
        });
    }

    public async getCompletion(prompt:string) {
      
    }

}
