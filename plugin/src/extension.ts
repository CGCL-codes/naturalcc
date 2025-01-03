// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
import { NccCompletionProvider } from './completion';
import { onConfigChange } from './config';

export const pluginName = 'nccc'
export var isActive: boolean
// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {

	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	vscode.window.showInformationMessage('Nccc is now active!');
	isActive = true
	const completor = new NccCompletionProvider();
	context.subscriptions.push(
		vscode.languages.registerInlineCompletionItemProvider(
			{pattern: '**'},
			completor
		),
		vscode.commands.registerCommand('nccc.enable', () => {
			vscode.window.showInformationMessage('Nccc is now active!');
			isActive = true
		  }),
		vscode.commands.registerCommand('nccc.disable', () => {
			vscode.window.showInformationMessage('Nccc is now inactive!');
			isActive = false
		  }),
	)
}

vscode.workspace.onDidChangeConfiguration((event) => {
	if(event.affectsConfiguration(pluginName)) {
		onConfigChange();
	}
})

// This method is called when your extension is deactivated
export function deactivate() {}
