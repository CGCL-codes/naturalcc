import { getOverlap } from './kmp'
import {
    InlineCompletionItem,
    InlineCompletionItemProvider,
    InlineCompletionList,
    InlineCompletionContext,
    CancellationToken,
    Position,
    Range,
    TextDocument
  } from 'vscode'
import { getCompletion } from './network';
import { getConfig } from './config';
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

        if(this.debounceTimer) {
          console.log('clear')
          clearTimeout(this.debounceTimer);
        }
        return new Promise((resolve) => {
          this.debounceTimer = setTimeout(async () => {
            console.log('trigger')
            const ret = await getCompletion(line)
            const completionRange = document.lineAt(position.line).range
            resolve([new InlineCompletionItem(ret, completionRange)])
          }, Number(getConfig().debounce));
        });
    }
}
