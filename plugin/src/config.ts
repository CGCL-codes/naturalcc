import { workspace } from 'vscode'
import { pluginName } from './extension'
export class NccConfig {
    public server: string
    public debounce: number
    public maxLength: number
    public topK: number
    public topP: number
    public temperature: number
    public contextLineCount: number
    constructor() {
        this.server = workspace.getConfiguration(pluginName).get('server') as string
        this.debounce = workspace.getConfiguration(pluginName).get('debounce') as number
        this.maxLength = workspace.getConfiguration(pluginName).get('maxLength') as number
        this.topK = workspace.getConfiguration(pluginName).get('topK') as number
        this.topP = workspace.getConfiguration(pluginName).get('topP') as number
        this.temperature = workspace.getConfiguration(pluginName).get('temperature') as number
        this.contextLineCount = workspace.getConfiguration(pluginName).get('contextLineCount') as number
    }
}

function updateConfig(c: NccConfig) {
    c.server = workspace.getConfiguration(pluginName).get('server') as string
    c.debounce = workspace.getConfiguration(pluginName).get('debounce') as number
    c.maxLength = workspace.getConfiguration(pluginName).get('maxLength') as number
    c.topK = workspace.getConfiguration(pluginName).get('topK') as number
    c.topP = workspace.getConfiguration(pluginName).get('topP') as number
    c.temperature = workspace.getConfiguration(pluginName).get('temperature') as number
    c.contextLineCount = workspace.getConfiguration(pluginName).get('contextLineCount') as number
}

var config: NccConfig

export function onConfigChange() {
    if(!config) {
        config = new NccConfig()
    }
    updateConfig(config)
}

export function getConfig(): NccConfig {
    onConfigChange()
    return config;
}

