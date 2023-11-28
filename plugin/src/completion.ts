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
            const ctx = getContext(document, position)
            const ret = await getCompletion(ctx)
            const completionRange = document.lineAt(position.line).range
            const compl = removeFirstNLines(ret, ctx.split('\n').length - 1)
            console.log(ctx)
            console.log(ctx.split('\n').length)
            console.log(compl)
            resolve([new InlineCompletionItem(compl, completionRange)])
          }, Number(getConfig().debounce));
        });
    }
}


function getContext(document: TextDocument, position: Position): string {
  const cnt = Math.min(getConfig().contextLineCount, position.line + 1)
  let ctx = ""
  for(let i = cnt - 1; i >= 0; i--) {
    const line = document.lineAt(position.line - i)
    if(!line.isEmptyOrWhitespace) {
      ctx += line.text
      if(i > 0) {
        ctx += '\n'
      }
    }
  }
  return ctx
}

function removeFirstNLines(str: string, n: number): string {
  const lines = str.split('\n');
  if(n >= lines.length) {
      return '';
  }
  if(n <= 0) {
    return str
  }
  return lines.slice(n).join('\n');
}